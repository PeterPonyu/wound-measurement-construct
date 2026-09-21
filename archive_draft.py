#!/usr/bin/env python3
"""Prepare a Zenodo software deposit; network operations require --execute.

An authenticated execution creates or resumes a draft and uploads one ZIP.
Review and publication take place in the returned Zenodo draft interface.
API reference: https://developers.zenodo.org/
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
import zipfile


def validate(metadata, archive):
    for field in ("title", "upload_type", "description", "creators", "version"):
        if not metadata.get(field):
            raise ValueError(f"Missing metadata: {field}")
    if metadata["upload_type"] != "software":
        raise ValueError("This uploader expects software metadata")
    if not zipfile.is_zipfile(archive):
        raise ValueError("Archive must be a valid ZIP")
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP integrity failed")
        manifests = [name for name in z.namelist() if name.endswith("/MANIFEST.sha256.json")]
        if len(manifests) != 1:
            raise ValueError("Archive must contain exactly one checksum manifest")
        manifest_name = manifests[0]
        prefix = manifest_name.removesuffix("MANIFEST.sha256.json")
        manifest = json.loads(z.read(manifest_name))
        expected = {prefix + name for name in manifest} | {manifest_name}
        actual = [name for name in z.namelist() if not name.endswith("/")]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("ZIP contains duplicate or unmanifested files")
        for name, digest in manifest.items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unexpected archive member path")
            if hashlib.sha256(z.read(prefix + name)).hexdigest() != digest:
                raise ValueError(f"Archived file checksum mismatch: {name}")
        packaged = json.loads(z.read(prefix + ".zenodo.json"))
        for field in ("title", "version", "creators", "related_identifiers"):
            if packaged.get(field) != metadata.get(field):
                raise ValueError(f"Metadata does not identify this archive: {field}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", type=Path)
    ap.add_argument("--metadata", type=Path, default=Path(__file__).with_name(".zenodo.json"))
    ap.add_argument("--execute", action="store_true", help="Create/resume an authenticated draft; never publish")
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--state", type=Path, help="Resume state for this study only")
    args = ap.parse_args()
    metadata = json.loads(args.metadata.read_text())
    validate(metadata, args.archive)
    summary = {"title": metadata["title"], "version": metadata["version"], "creators": metadata["creators"], "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(), "bytes": args.archive.stat().st_size, "license_specified": bool(metadata.get("license")), "mode": "offline validation"}
    if not args.execute:
        print(json.dumps(summary, indent=2))
        return
    if not metadata.get("license"):
        raise ValueError("Specify the author-approved distribution license in the metadata before deposit")
    token = os.environ.get("ZENODO_SANDBOX_TOKEN" if args.sandbox else "ZENODO_ACCESS_TOKEN")
    if not token:
        raise ValueError("Set the appropriate Zenodo token in the environment")
    host = "sandbox.zenodo.org" if args.sandbox else "zenodo.org"
    base = f"https://{host}/api/deposit/depositions"
    identity = hashlib.sha256(metadata["title"].encode()).hexdigest()[:16]
    if args.state is None:
        args.state = Path(__file__).parent / ".archive-state" / f"{host}-{identity}.json"
    args.state.parent.mkdir(parents=True, exist_ok=True)

    def api(method, url, payload=None, binary=None):
        if urlparse(url).scheme != "https" or urlparse(url).hostname != host:
            raise ValueError("Unexpected archive API origin")
        data = binary if binary is not None else json.dumps(payload).encode() if payload is not None else None
        headers = {"Authorization": "Bearer " + token, "Content-Type": "application/octet-stream" if binary is not None else "application/json"}
        try:
            with urlopen(Request(url, data=data, headers=headers, method=method), timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"Zenodo returned HTTP {exc.code}; existing draft state is retained") from None

    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state["host"] != host:
            raise ValueError("State belongs to a different archive host")
        if state.get("study_identity") != identity:
            raise ValueError("State belongs to a different study; use its own state file")
        record = api("GET", base + "/" + str(state["deposition_id"]))
        if record.get("metadata", {}).get("title") not in (None, metadata["title"]):
            raise ValueError("The existing deposit belongs to a different study")
    else:
        record = api("POST", base, {})
        state = {"host": host, "study_identity": identity, "deposition_id": record["id"]}
        args.state.write_text(json.dumps(state, indent=2) + "\n")
    if record.get("submitted"):
        raise ValueError("The record is already submitted; create a new version in Zenodo")
    record = api("PUT", base + "/" + str(record["id"]), {"metadata": metadata})
    payload = args.archive.read_bytes()
    file_result = api("PUT", record["links"]["bucket"] + "/" + quote(args.archive.name), binary=payload)
    expected = hashlib.md5(payload).hexdigest()
    if file_result.get("checksum", "").removeprefix("md5:") != expected:
        raise RuntimeError("Uploaded ZIP checksum mismatch")
    state.update({"draft_url": record["links"].get("html"), "archive_sha256": summary["archive_sha256"], "reserved_doi": record.get("metadata", {}).get("prereserve_doi", {}).get("doi"), "published": False})
    args.state.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
