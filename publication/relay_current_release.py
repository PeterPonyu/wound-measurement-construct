#!/usr/bin/env python3
"""Copy this product's hash-verified public release to its current Zenodo draft."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
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
USER_AGENT = 'ScientificReportsSourceRelease/1.1 (+https://github.com/' + CONFIG['repository'] + '; fuzeyu09@gmail.com)'
WORKERS = 2

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
               'User-Agent': USER_AGENT,
               'Content-Type': 'application/octet-stream' if binary is not None else 'application/json'}
    if method == 'GET':
        url += ('&' if parsed.query else '?') + 'fresh=' + str(time.time_ns())
    try:
        with API.open(Request(url, data=body, headers=headers, method=method), timeout=600) as response:
            payload = response.read()
            return json.loads(payload) if payload else None
    except HTTPError as exc:
        raise RuntimeError('Zenodo ' + method + ' ' + parsed.path + ' HTTP ' + str(exc.code)) from None
    except (URLError, TimeoutError, ConnectionError, ssl.SSLError):
        raise RuntimeError('Zenodo connection failed; publication state must be rechecked') from None

def retry(method, url, data=None, binary=None):
    for n in range(4):
        try: return api(method, url, data, binary)
        except RuntimeError:
            if n == 3: raise
            time.sleep(5 * (n + 1))

def public_probe(url):
    """An unpublished record is an ordinary 404; transport errors leave state unknown."""
    for attempt in range(6):
        request = Request(url + '?fresh=' + str(time.time_ns()),
                          headers={'User-Agent': USER_AGENT, 'Cache-Control': 'no-cache',
                                   'Accept': 'application/json'})
        try:
            with urlopen(request, timeout=90) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise RuntimeError('Public Zenodo record HTTP ' + str(exc.code)) from None
        except (URLError, TimeoutError, ConnectionError, ssl.SSLError):
            pass
        if attempt < 5:
            time.sleep(min(60, 5 * (2 ** attempt)))
    raise RuntimeError('Public Zenodo record could not be read; publication state is unknown')


def verified_download(url, target, item):
    """Resume verified files; replace an incomplete download only after full hash checks."""
    if (target.exists() and target.stat().st_size == item['bytes']
            and digest(target) == item['sha256'] and digest(target, 'md5') == item['md5']):
        return
    partial = target.with_name(target.name + '.partial')
    for attempt in range(6):
        try:
            request = Request(url, headers={'User-Agent': USER_AGENT, 'Cache-Control': 'no-cache'})
            with urlopen(request, timeout=180) as response, partial.open('wb') as stream:
                for block in iter(lambda: response.read(1 << 20), b''):
                    stream.write(block)
            if (partial.stat().st_size != item['bytes'] or digest(partial) != item['sha256']
                    or digest(partial, 'md5') != item['md5']):
                raise RuntimeError(item['name'] + ': downloaded payload differs from sealed release')
            partial.replace(target)
            return
        except HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise RuntimeError(item['name'] + ': download HTTP ' + str(exc.code)) from None
        except (URLError, TimeoutError, ConnectionError, ssl.SSLError):
            pass
        if attempt < 5:
            time.sleep(min(60, 5 * (2 ** attempt)))
    raise RuntimeError(item['name'] + ': download unavailable after bounded retries')


def download(item):
    url = item['url']; parts = urlsplit(url)
    assert parts.scheme == 'https' and parts.hostname == 'github.com'
    assert parts.path.startswith('/' + CONFIG['repository'] + '/releases/download/' + CONFIG['tag'] + '/')
    name = item['name']; assert name == Path(name).name
    path = ROOT / name
    verified_download(url, path, item)
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

local = {} if '--verify-public' in sys.argv[2:] else {item['name']: download(item) for item in CONFIG['files']}
if '--verify-public' not in sys.argv[2:]:
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

def remove_interrupted(entry):
    assert entry.get('status') == 'pending' and entry.get('size') in (None, 0)
    assert entry['links']['self'].startswith(files_url + '/')
    for attempt in range(4):
        current = list_entries().get(entry['key'])
        if not current or current.get('status') == 'completed': return
        try: api('DELETE', current['links']['self'])
        except RuntimeError: pass
        for delay in (2, 5, 10):
            if entry['key'] not in list_entries(): return
            time.sleep(delay)
    raise RuntimeError(entry['key'] + ': interrupted upload could not be cleared; preserve unpublished state')

def stage_release():
    # Incomplete placeholders can prevent Zenodo from serializing draft metadata.
    # Remove only expected, zero-byte pending entries before reading metadata.
    interrupted = [entry for entry in list_entries().values() if entry.get('status') == 'pending']
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(remove_interrupted, interrupted))
    draft = retry('GET', 'https://zenodo.org/api/deposit/depositions/' + record_id)
    assert draft['metadata']['version'] == CONFIG['version']
    if draft.get('submitted') is True:
        return  # Continue below with public-file verification, without a mutation.
    assert draft.get('submitted') is False
    def upload_one(pair):
        name, item = pair
        entry = list_entries().get(name)
        if good(entry, item):
            print(name, 'already verified', flush=True); return
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
                if not entry:
                    if attempt == 3: raise
                    time.sleep(10 * (attempt + 1))
                    continue
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


    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(upload_one, by_name.items()))

    entries = list_entries()
    assert set(entries) == set(by_name) and all(good(entries[n], item) for n, item in by_name.items())
    try:
        api('POST', 'https://zenodo.org/api/deposit/depositions/' + record_id + '/actions/publish')
    except RuntimeError:
        # An ambiguous publish reply is followed by a public read, never blind repetition.
        pass

public = public_probe(base)
if public is None:
    if '--verify-public' in sys.argv[2:]:
        raise RuntimeError('The requested record is still unpublished')
    try:
        stage_release()
    except RuntimeError as exc:
        print(str(exc) + '; completed draft files are preserved for resume', flush=True)
    public = public_probe(base)
    if public is None:
        raise RuntimeError('Draft remains unpublished; resume verified staging rather than treating 404 as a publication failure')
assert public['doi'] == CONFIG['doi'] and public['metadata']['version'] == CONFIG['version']
assert {x['key'] for x in public['files']} == set(by_name)
published = {x['key']: x for x in public['files']}
verified = []
public_root = Path('published-downloads'); public_root.mkdir(exist_ok=True)
def download_public(pair):
    name, item = pair
    remote = published[name]
    assert remote['size'] == item['bytes'] and remote['checksum'].removeprefix('md5:') == item['md5']
    target = public_root / name
    url = 'https://zenodo.org/records/' + record_id + '/files/' + quote(name) + '?download=1&fresh=' + str(time.time_ns())
    verified_download(url, target, item)
    assert target.stat().st_size == item['bytes'] and digest(target) == item['sha256']
    assert digest(target, 'md5') == item['md5']
    print(name, 'public Zenodo download verified', flush=True)
    return {'name': name, 'bytes': item['bytes'], 'sha256': item['sha256'], 'md5': item['md5']}

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    verified = list(pool.map(download_public, by_name.items()))
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
