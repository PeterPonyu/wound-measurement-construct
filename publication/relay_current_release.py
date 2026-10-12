#!/usr/bin/env python3
"""Copy this product's hash-verified public release to its current Zenodo draft."""
import hashlib
import ssl
import json
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler, urlopen
import zipfile

CONFIG = json.loads(Path(sys.argv[1]).read_text())
ROOT = Path('verified-release'); ROOT.mkdir(exist_ok=True)

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

API = build_opener(ProxyHandler({}), NoRedirect())

def digest(path, algorithm='sha256'):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()

def api(method, url, data=None, binary=None):
    parsed = urlsplit(url)
    assert parsed.scheme == 'https' and parsed.hostname == 'zenodo.org' and parsed.path.startswith('/api/')
    token = (os.environ.get('ZENODO_ACCESS_TOKEN') or '').strip()
    if not token:
        raise RuntimeError('The temporary publication secret is not configured')
    body = binary if binary is not None else json.dumps(data).encode() if data is not None else None
    headers = {'Authorization': 'Bearer ' + token,
               'Accept': 'application/json',
               'Cache-Control': 'no-cache',
               'User-Agent': 'scientific-reports-release-transfer/1.0',
               'Content-Type': 'application/octet-stream' if binary is not None else 'application/json'}
    if method == 'GET':
        url += ('&' if parsed.query else '?') + 'fresh=' + str(time.time_ns())
    try:
        with API.open(Request(url, data=body, headers=headers, method=method), timeout=600) as response:
            payload = response.read()
            return json.loads(payload) if payload else None
    except HTTPError as exc:
        raise RuntimeError('Zenodo ' + method + ' HTTP ' + str(exc.code)) from None
    except (URLError, TimeoutError, ConnectionError, ssl.SSLError):
        raise RuntimeError('Zenodo connection failed; publication state must be rechecked') from None

def retry(method, url, data=None, binary=None):
    for n in range(4):
        try: return api(method, url, data, binary)
        except RuntimeError:
            if n == 3: raise
            time.sleep(5 * (n + 1))

def download(item):
    url = item['url']; parts = urlsplit(url)
    assert parts.scheme == 'https' and parts.hostname == 'github.com'
    assert parts.path.startswith('/' + CONFIG['repository'] + '/releases/download/' + CONFIG['tag'] + '/')
    name = item['name']; assert name == Path(name).name
    path = ROOT / name
    if not path.exists() or digest(path) != item['sha256']:
        partial = path.with_suffix(path.suffix + '.partial')
        with urlopen(url, timeout=180) as response, partial.open('wb') as target:
            for block in iter(lambda: response.read(1 << 20), b''): target.write(block)
        assert partial.stat().st_size == item['bytes'] and digest(partial) == item['sha256'], name
        partial.replace(path)
    assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
    assert digest(path, 'md5') == item['md5']
    print(name, 'downloaded and hash verified', flush=True)
    return path

if '--preflight' in sys.argv[2:] or '--cleanup-pending' in sys.argv[2:]:
    record_id = str(CONFIG['record_id'])
    assert record_id in ('23308073', '23308074', '23308077')
    base = 'https://zenodo.org/api/records/' + record_id
    files_url = base + '/draft/files'
    entries = retry('GET', files_url)['entries']
    assert {x['key'] for x in entries}.issubset({x['name'] for x in CONFIG['files']})
    if '--cleanup-pending' in sys.argv[2:]:
        for entry in entries:
            if entry.get('status') != 'pending': continue
            assert entry.get('size') in (None, 0)
            assert entry['links']['self'].startswith(files_url + '/')
            try: api('DELETE', entry['links']['self'])
            except RuntimeError:
                remaining = {x['key'] for x in retry('GET', files_url)['entries']}
                if entry['key'] in remaining: raise
            print(entry['key'], 'interrupted private-draft placeholder removed', flush=True)
        entries = retry('GET', files_url)['entries']
    print(json.dumps({'status': 'cloud-connection-verified', 'record_id': CONFIG['record_id'],
                      'files': len(entries), 'completed': sum(x.get('status') == 'completed' for x in entries)}), flush=True)
    raise SystemExit(0)

local = {item['name']: download(item) for item in CONFIG['files']}
complete = ROOT / CONFIG['canonical_archive_name']
with complete.open('wb') as target:
    for name in CONFIG['archive_parts']:
        with local[name].open('rb') as part:
            for block in iter(lambda: part.read(1 << 20), b''): target.write(block)
assert digest(complete) == CONFIG['canonical_archive_sha256']
with zipfile.ZipFile(complete) as archive:
    assert archive.testzip() is None
print('All source payloads and canonical ZIP SHA256/CRC verified', flush=True)

record_id = str(CONFIG['record_id'])
assert record_id in ('23308073', '23308074', '23308077')
base = 'https://zenodo.org/api/records/' + record_id
files_url = base + '/draft/files'
by_name = {item['name']: item for item in CONFIG['files']}

def good(entry, item):
    return entry and entry.get('status') == 'completed' and entry.get('size') == item['bytes'] and (entry.get('checksum') or '').removeprefix('md5:') == item['md5']

def list_entries():
    entries = {x['key']: x for x in retry('GET', files_url)['entries']}
    assert set(entries).issubset(by_name), 'Unexpected file: preserve it and stop'
    return entries

def stage_release():
    # Incomplete placeholders can prevent Zenodo from serializing draft metadata.
    # Remove only expected, zero-byte pending entries before reading metadata.
    for entry in list_entries().values():
        if entry.get('status') != 'pending': continue
        assert entry.get('size') in (None, 0)
        assert entry['links']['self'].startswith(files_url + '/')
        try: api('DELETE', entry['links']['self'])
        except RuntimeError:
            if entry['key'] in list_entries(): raise
    draft = retry('GET', 'https://zenodo.org/api/deposit/depositions/' + record_id)
    assert draft['metadata']['version'] == CONFIG['version']
    if draft.get('submitted') is True:
        return  # Continue below with public-file verification, without a mutation.
    assert draft.get('submitted') is False
    entries = list_entries()
    for name, item in by_name.items():
        entry = entries.get(name)
        if good(entry, item):
            print(name, 'already verified', flush=True); continue
        for attempt in range(4):
            entry = list_entries().get(name)
            if good(entry, item): break
            if entry:
                # Remove only an interrupted file in this fixed, unpublished draft.
                try: api('DELETE', entry['links']['self'])
                except RuntimeError:
                    if list_entries().get(name): raise
            try:
                initialized = api('POST', files_url, [{'key': name}])
                entry = next(x for x in initialized['entries'] if x['key'] == name)
            except RuntimeError:
                # Initialization may have committed despite a lost response.
                entry = list_entries().get(name)
                if not entry: raise
            try:
                api('PUT', entry['links']['content'], binary=local[name].read_bytes())
                api('POST', entry['links']['commit'])
            except RuntimeError:
                current = list_entries().get(name)
                if good(current, item): break
                # A lost upload response can precede a successful commit.
                if current:
                    try: api('POST', current['links']['commit'])
                    except RuntimeError: pass
            if good(list_entries().get(name), item): break
            if attempt == 3: raise RuntimeError(name + ': upload not verified; draft remains unpublished')
            time.sleep(10 * (attempt + 1))
        assert good(list_entries().get(name), item), name
        print(name, 'committed with verified size and MD5', flush=True)

    entries = list_entries()
    assert set(entries) == set(by_name) and all(good(entries[n], item) for n, item in by_name.items())
    try:
        api('POST', 'https://zenodo.org/api/deposit/depositions/' + record_id + '/actions/publish')
    except RuntimeError:
        # An ambiguous publish reply is followed by a public read, never blind repetition.
        pass

try:
    stage_release()
except RuntimeError as exc:
    # A failed or ambiguous transfer never proves publication. The exact public
    # DOI, version, file set and downloaded payloads must still pass below.
    print(str(exc) + '; checking the public record after an ambiguous or resumed transfer', flush=True)
with urlopen(base + '?fresh=' + str(time.time_ns()), timeout=90) as response: public = json.load(response)
assert public['doi'] == CONFIG['doi'] and public['metadata']['version'] == CONFIG['version']
assert {x['key'] for x in public['files']} == set(by_name)
published = {x['key']: x for x in public['files']}
verified = []
public_root = Path('published-downloads'); public_root.mkdir(exist_ok=True)
for name, item in by_name.items():
    remote = published[name]
    assert remote['size'] == item['bytes'] and remote['checksum'].removeprefix('md5:') == item['md5']
    target = public_root / name
    url = 'https://zenodo.org/records/' + record_id + '/files/' + quote(name) + '?download=1&fresh=' + str(time.time_ns())
    with urlopen(url, timeout=180) as response, target.open('wb') as stream:
        for block in iter(lambda: response.read(1 << 20), b''): stream.write(block)
    assert target.stat().st_size == item['bytes'] and digest(target) == item['sha256']
    assert digest(target, 'md5') == item['md5']
    verified.append({'name': name, 'bytes': item['bytes'], 'sha256': item['sha256'], 'md5': item['md5']})
    print(name, 'public Zenodo download verified', flush=True)
public_zip = public_root / CONFIG['canonical_archive_name']
with public_zip.open('wb') as target:
    for name in CONFIG['archive_parts']:
        with (public_root / name).open('rb') as part:
            for block in iter(lambda: part.read(1 << 20), b''): target.write(block)
assert digest(public_zip) == CONFIG['canonical_archive_sha256']
with zipfile.ZipFile(public_zip) as archive: assert archive.testzip() is None
receipt = {'status': 'published-and-download-verified', 'record_id': CONFIG['record_id'], 'doi': CONFIG['doi'],
           'version': CONFIG['version'], 'canonical_archive_sha256': CONFIG['canonical_archive_sha256'],
           'payload_files': len(by_name), 'publication_url': 'https://zenodo.org/records/' + record_id,
           'public_download_verification': verified, 'canonical_zip_crc': 'PASS'}
Path('publication-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
