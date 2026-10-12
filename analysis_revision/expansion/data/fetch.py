"""Download planning, approval-gated execution and checksum verification."""
from __future__ import annotations

import hashlib
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..common.io import atomic_write_json, read_json, sha256_file


class ApprovalError(PermissionError):
    """Downloads need an approval file naming every dataset in the batch."""


@dataclass(frozen=True)
class FetchItem:
    dataset_id: str
    url: str
    destination: str
    size_label: str
    md5: str | None = None
    bytes: int | None = None


class IncompleteDownloadError(IOError):
    """A transfer ended early or failed its size or checksum; nothing was accepted."""


def md5_file(path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_fetch(entries, *, ids=None) -> list:
    wanted = set(ids) if ids is not None else None
    items = []
    for entry in entries:
        if entry.status != "to_fetch" or (wanted is not None and entry.id not in wanted):
            continue
        files = list(entry.source.get("files", [])) + list(entry.extra.get("extra_files", []))
        for file in files:
            url = file.get("url") or entry.source["base_url"] + file["name"]
            items.append(FetchItem(entry.id, url, f"{entry.local_path}/{file['name']}", file.get("size_label", ""),
                                   file.get("md5"), file.get("bytes")))
    return items


def read_approval(path, ids) -> dict:
    path = Path(path)
    if not path.exists():
        raise ApprovalError(f"approval file {path} does not exist")
    approval = read_json(path)
    for key in ("approved_by", "approved_utc", "dataset_ids"):
        if not approval.get(key):
            raise ApprovalError(f"approval file lacks {key}")
    missing = sorted(set(ids) - set(approval["dataset_ids"]))
    if missing:
        raise ApprovalError(f"approval does not cover {missing}")
    return approval


def _download(item: FetchItem, target: Path, opener, chunk: int) -> None:
    """Stream to ``<target>.part``; accept only a complete transfer that matches the declared size and md5."""
    partial = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(item.url, headers={"User-Agent": "expansion-fetch/0.1"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with opener(request) as response:
        status = getattr(response, "status", None)
        headers = getattr(response, "headers", None)
        declared = headers.get("Content-Length") if headers is not None else None
        mode = "ab" if offset and status == 206 else "wb"
        received = 0
        with open(partial, mode) as sink:
            while True:
                block = response.read(chunk)
                if not block:
                    break
                sink.write(block)
                received += len(block)
    if declared is not None and received != int(declared):
        raise IncompleteDownloadError(f"{item.url}: received {received} of {declared} bytes; kept {partial.name} for resume")
    if item.bytes is not None and partial.stat().st_size != int(item.bytes):
        raise IncompleteDownloadError(f"{item.url}: size {partial.stat().st_size} differs from registry {item.bytes}")
    if item.md5 is not None and md5_file(partial) != item.md5:
        partial.unlink()
        raise IncompleteDownloadError(f"{item.url}: md5 mismatch; partial file removed")
    partial.replace(target)


def execute_fetch(items, *, root, approval_path, manifest_path=None, opener=None, chunk: int = 1 << 20) -> dict:
    """Download approved files, recording sha256 at first retrieval (trust on first use)."""
    root = Path(root)
    approval = read_approval(approval_path, {item.dataset_id for item in items})
    manifest_path = Path(manifest_path) if manifest_path else root / "data/raw/fetch_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    opener = opener or urllib.request.urlopen
    summary = {"downloaded": 0, "skipped_verified": 0, "approval": approval["approved_utc"]}
    for item in items:
        target = root / item.destination
        record = manifest.get(item.dataset_id, {}).get(Path(item.destination).name)
        if target.exists() and record and sha256_file(target) == record["sha256"]:
            summary["skipped_verified"] += 1
            continue
        _download(item, target, opener, chunk)
        manifest.setdefault(item.dataset_id, {})[Path(item.destination).name] = {
            "url": item.url, "sha256": sha256_file(target), "bytes": target.stat().st_size,
            "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "approved_by": approval["approved_by"]}
        atomic_write_json(manifest_path, manifest)
        summary["downloaded"] += 1
    return summary


def verify_fetch(root, manifest_path) -> list:
    root = Path(root)
    problems = []
    for dataset_id, files in read_json(manifest_path).items():
        for name, record in files.items():
            matches = list(root.glob(f"data/raw/**/{name}"))
            if not matches:
                problems.append(f"{dataset_id}/{name}: missing")
            elif all(sha256_file(p) != record["sha256"] for p in matches):
                problems.append(f"{dataset_id}/{name}: checksum mismatch")
    return problems
