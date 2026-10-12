"""Fail-closed, approval-bound payload transfer.

This module is intentionally separate from :mod:`expansion.data.fetch`.
The older fetcher is retained for historical compatibility and unit fixtures;
this layer is the only transfer path used by the 2026 launch kit.  It binds a
transfer to an exact URL, destination, version, byte count, SHA-256 and strong
HTTP validator.  It never overwrites an existing payload and writes only to a
fresh quarantine root.

The implementation is deliberately dependency-free so the same code can run
inside the future portable bundle.  It accepts an injected opener in tests,
but the production opener rejects redirects before contacting another URL.
"""
from __future__ import annotations

import hashlib
import json
import os
from queue import Empty, Queue
import re
import threading
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

from ..common.io import atomic_write_json, read_json, sha256_file


class TransferApprovalError(PermissionError):
    """The approval receipt is absent, malformed, stale or mismatched."""


class TransferIntegrityError(IOError):
    """A server response or local transfer state failed a safety check."""


class TransferPathError(ValueError):
    """A destination is not an exact safe relative path."""


@dataclass(frozen=True)
class TransferSpec:
    dataset_id: str
    url: str
    destination: str
    version: str
    expected_bytes: int
    sha256: str
    validator_kind: str
    validator_value: str

    def as_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "url": self.url,
            "destination": self.destination,
            "version": self.version,
            "expected_bytes": self.expected_bytes,
            "sha256": self.sha256,
            "validator": {self.validator_kind: self.validator_value},
        }


def _item_key(spec: TransferSpec) -> str:
    """Stable identity for one approved file (one dataset may have many files)."""
    return f"{spec.dataset_id}::{spec.destination}"


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_RANGE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.I)
_ALLOWED_APPROVAL_KEYS = {
    "schema_version", "kind", "scope", "approved", "approved_by", "approved_utc",
    "approval_id", "items", "notes",
}


def _strict_json(path: Path) -> dict:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise TransferApprovalError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except FileNotFoundError as error:
        raise TransferApprovalError(f"approval file does not exist: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise TransferApprovalError(f"invalid approval JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TransferApprovalError("approval must be a JSON object")
    return payload


def _safe_destination(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise TransferPathError(f"destination is not a safe POSIX relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or str(path) != value:
        raise TransferPathError(f"destination is not normalized and relative: {value!r}")
    if not path.parts or any(not part for part in path.parts):
        raise TransferPathError(f"destination contains an empty component: {value!r}")
    return value


def _safe_url(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("https://"):
        raise TransferApprovalError(f"transfer URL must use HTTPS: {value!r}")
    parsed = urllib.parse.urlsplit(value)
    if not parsed.hostname or parsed.username or parsed.password or parsed.fragment \
            or any(ord(c) <= 32 for c in value):
        raise TransferApprovalError("transfer URL must have an exact host, no credentials/fragment/control characters")
    return value


def _parse_validator(value: object) -> tuple[str, str]:
    if not isinstance(value, dict) or set(value) not in ({"etag"}, {"last_modified"}):
        raise TransferApprovalError("validator must contain exactly etag or last_modified")
    kind, validator = next(iter(value.items()))
    if not isinstance(validator, str) or not validator.strip():
        raise TransferApprovalError("validator value must be a non-empty string")
    if kind == "etag" and validator.lstrip().startswith("W/"):
        raise TransferApprovalError("weak ETags cannot be used as an If-Range validator")
    return kind, validator


def _parse_spec(item: object, *, context: str) -> TransferSpec:
    if not isinstance(item, dict):
        raise TransferApprovalError(f"{context} must be an object")
    required = {"dataset_id", "url", "destination", "version", "expected_bytes", "sha256", "validator"}
    if set(item) != required:
        raise TransferApprovalError(f"{context} keys must be exactly {sorted(required)}")
    dataset_id = item["dataset_id"]
    version = item["version"]
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise TransferApprovalError(f"{context}.dataset_id is required")
    if not isinstance(version, str) or not version.strip():
        raise TransferApprovalError(f"{context}.version is required")
    try:
        expected_bytes = int(item["expected_bytes"])
    except (TypeError, ValueError) as error:
        raise TransferApprovalError(f"{context}.expected_bytes must be an integer") from error
    if type(item["expected_bytes"]) is not int or expected_bytes <= 0:
        raise TransferApprovalError(f"{context}.expected_bytes must be a positive integer")
    digest = item["sha256"]
    if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        raise TransferApprovalError(f"{context}.sha256 must be a lowercase SHA-256 digest")
    kind, validator = _parse_validator(item["validator"])
    return TransferSpec(dataset_id, _safe_url(item["url"]), _safe_destination(item["destination"]),
                        version, expected_bytes, digest, kind, validator)


def read_transfer_approval(path: str | Path) -> tuple[dict, list[TransferSpec]]:
    """Read a strict, approved transfer receipt.

    Planning manifests intentionally fail this parser: they have a different
    ``kind`` and do not contain a signed, item-level payload contract.
    """
    path = Path(path)
    approval = _strict_json(path)
    unknown = set(approval) - _ALLOWED_APPROVAL_KEYS
    if unknown:
        raise TransferApprovalError(f"approval has unknown keys: {sorted(unknown)}")
    required = {"schema_version", "kind", "scope", "approved", "approved_by", "approved_utc",
                "approval_id", "items"}
    missing = required - set(approval)
    if missing:
        raise TransferApprovalError(f"approval lacks {sorted(missing)}")
    if approval["schema_version"] != 1 or approval["kind"] != "sdd_transfer_approval":
        raise TransferApprovalError("file is not a schema-version-1 sdd_transfer_approval")
    if approval["scope"] != "payload_transfer":
        raise TransferApprovalError("approval scope must be payload_transfer")
    if approval["approved"] is not True:
        raise TransferApprovalError("transfer approval is not explicitly approved=true")
    for key in ("approved_by", "approval_id"):
        if not isinstance(approval[key], str) or not approval[key].strip():
            raise TransferApprovalError(f"approval.{key} must be non-empty")
    stamp = approval["approved_utc"]
    try:
        parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise TransferApprovalError("approval.approved_utc must be ISO-8601") from error
    if parsed.tzinfo is None or parsed > datetime.now(timezone.utc):
        raise TransferApprovalError("approval timestamp must have a timezone and not be in the future")
    if not isinstance(approval["items"], list) or not approval["items"]:
        raise TransferApprovalError("approval.items must be a non-empty list")
    specs = [_parse_spec(item, context=f"approval.items[{i}]") for i, item in enumerate(approval["items"])]
    if len({s.destination for s in specs}) != len(specs):
        raise TransferApprovalError("approval contains duplicate destinations")
    if len({_item_key(s) for s in specs}) != len(specs):
        raise TransferApprovalError("approval contains duplicate dataset/destination items")
    return approval, specs


def _validate_requested(requested: Iterable[TransferSpec]) -> list[TransferSpec]:
    specs = list(requested)
    if not specs:
        raise TransferApprovalError("no transfer items were requested")
    for spec in specs:
        if not isinstance(spec, TransferSpec):
            raise TypeError("requested items must be TransferSpec instances")
        if _parse_spec(spec.as_dict(), context="requested item") != spec:
            raise TransferApprovalError("requested item is not canonical")
    if len({s.destination for s in specs}) != len(specs) or len({_item_key(s) for s in specs}) != len(specs):
        raise TransferApprovalError("requested transfer items contain duplicate destinations or items")
    return specs


def _bind_approval(approval_specs: list[TransferSpec], requested: list[TransferSpec]) -> None:
    expected = {_item_key(s): s.as_dict() for s in requested}
    actual = {_item_key(s): s.as_dict() for s in approval_specs}
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise TransferApprovalError(f"approval does not exactly bind requested items; missing={missing}, extra={extra}")


def _assert_fresh_root(root: Path) -> None:
    root = Path(os.path.abspath(root))
    for parent in [root, *root.parents]:
        if parent.is_symlink():
            raise TransferPathError(f"symlink in transfer root ancestry: {parent}")
        if (parent / "FINAL_STATUS.json").exists():
            raise TransferPathError(f"sealed historical root cannot receive transfers: {parent}")
    if root.exists():
        if not root.is_dir():
            raise TransferPathError(f"transfer root must be a real directory: {root}")
        # The root may be created by the caller, but a payload destination must
        # not already exist.  We do not require the whole root to be empty so a
        # receipt can be resumed without deleting unrelated planning evidence.
    else:
        root.mkdir(parents=True)
    for parent in [root, *root.parents]:
        if parent.is_symlink():
            raise TransferPathError(f"symlink in transfer root ancestry: {parent}")


def _target(root: Path, destination: str) -> Path:
    _safe_destination(destination)
    root = Path(os.path.abspath(root))
    path = root / Path(destination)
    if path.is_symlink():
        raise TransferPathError(f"destination is an existing symlink: {destination}")
    for parent in [path.parent, *path.parent.parents]:
        if parent.is_symlink():
            raise TransferPathError(f"destination has a symlinked parent: {destination}")
    resolved_root = root.resolve()
    try:
        path.parent.resolve().relative_to(resolved_root)
    except ValueError as error:
        raise TransferPathError(f"destination escapes transfer root: {destination}") from error
    return path


def _header(headers, name: str) -> str | None:
    if headers is None:
        return None
    wanted = name.lower()
    value = None
    # urllib's HTTPMessage is case-insensitive, while injected test doubles
    # and a few lightweight clients use ordinary dictionaries.  Normalize the
    # lookup here so both paths enforce the same validator/range contract.
    for key in (name, name.lower(), name.title()):
        try:
            value = headers.get(key)
        except AttributeError:
            value = None
        if value is not None:
            break
    if value is None and hasattr(headers, "items"):
        for key, candidate in headers.items():
            if str(key).lower() == wanted:
                value = candidate
                break
    return str(value).strip() if value is not None else None


def _integer_header(headers, name: str) -> int:
    """Return a required non-negative integer response header.

    A transfer is not accepted on the basis of a streamed byte count alone:
    the server must also declare the exact body length for the response mode.
    Keeping this parser strict makes missing, malformed and negative lengths
    fail closed before any payload can be published.
    """
    value = _header(headers, name)
    if value is None:
        raise TransferIntegrityError(f"response lacks required {name} header")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise TransferIntegrityError(f"response has invalid {name}: {value!r}") from error
    if number < 0:
        raise TransferIntegrityError(f"response has invalid negative {name}: {value!r}")
    return number


class _TimedReader:
    """Read response chunks under an independent wall-clock deadline.

    A socket timeout limits one socket operation; it does not bound a fake or
    custom reader that never returns.  The daemon reader lets the caller stop
    waiting and retain only the resumable ``.part`` state.  The response is
    still closed by the surrounding context manager, and the daemon cannot
    publish a final target.
    """

    def __init__(self, response):
        self._response = response
        self._requests: Queue[int | None] = Queue(maxsize=1)
        self._results: Queue[tuple[bool, object]] = Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="sdd-transfer-reader", daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                size = self._requests.get()
            except Exception:
                return
            if size is None or self._stop.is_set():
                return
            try:
                self._results.put((True, self._response.read(size)))
            except BaseException as error:
                self._results.put((False, error))

    def read(self, size: int, deadline: float):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("transfer wall timeout reached before response read")
        self._requests.put(size, timeout=remaining)
        try:
            ok, value = self._results.get(timeout=remaining)
        except Empty as error:
            self._stop.set()
            _abort_response_socket(self._response)
            raise TimeoutError("transfer wall timeout reached during response read") from error
        if not ok:
            if isinstance(value, BaseException):
                raise value
            raise TransferIntegrityError("reader returned a non-exception error")
        return value

    def close(self):
        self._stop.set()
        try:
            self._requests.put_nowait(None)
        except Exception:
            pass


def _abort_response_socket(response) -> None:
    """Interrupt stdlib HTTPResponse reads without acquiring its buffer lock."""
    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


@contextmanager
def _bounded_response(response, deadline: float):
    """Never wait synchronously for a buffer lock held by an expired read.

    Production sockets are interrupted on failure. Closing runs on a daemon
    with a bounded wait, so a hostile injected close/read cannot hold the caller
    indefinitely or publish a payload. A stuck custom reader may retain its
    daemon until it returns; this is not process-level resource isolation.
    """
    try:
        yield response
    except BaseException:
        _abort_response_socket(response)
        raise
    finally:
        def close():
            try:
                if callable(getattr(response, "close", None)):
                    response.close()
                elif callable(getattr(response, "__exit__", None)):
                    response.__exit__(None, None, None)
            except Exception:
                pass
        closer = threading.Thread(target=close, name="sdd-transfer-close", daemon=True)
        closer.start()
        closer.join(timeout=max(0.0, min(.05, deadline - time.monotonic())))


def _call_opener(opener: Callable, request, timeout: float):
    if opener is _open_without_redirects:
        return opener(request, timeout=timeout)
    return opener(request)


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        raise TransferIntegrityError(f"redirected request rejected before contacting {new_url}")


def _open_without_redirects(request, *, timeout):
    # Approved NCBI/Zenodo intake is direct. Desktop/Turbo proxy inheritance is
    # not part of the exact transfer contract and must not affect it silently.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _RejectRedirects()).open(request, timeout=timeout)


def _load_receipt(path: Path, *, allow_missing: bool = False) -> dict:
    if not path.exists() and allow_missing:
        return {"schema_version": 1, "kind": "sdd_transfer_receipt", "items": {}}
    try:
        payload = _strict_json(path)
    except TransferApprovalError as error:
        raise TransferIntegrityError(f"invalid or missing transfer receipt: {error}") from error
    if payload.get("schema_version") != 1 or payload.get("kind") != "sdd_transfer_receipt" \
            or not isinstance(payload.get("items"), dict):
        raise TransferIntegrityError(f"invalid transfer receipt: {path}")
    if not payload.get("approved_items") or not isinstance(payload["approved_items"], list):
        raise TransferIntegrityError("receipt has no complete approved item contract")
    try:
        specs = [_parse_spec(item, context="receipt.approved_items") for item in payload["approved_items"]]
        _validate_requested(specs)
    except (TransferApprovalError, TransferPathError, TypeError) as error:
        raise TransferIntegrityError(f"invalid receipt contract: {error}") from error
    expected = {_item_key(spec): spec for spec in specs}
    if not set(payload["items"]) <= set(expected):
        raise TransferIntegrityError("receipt contains unapproved item identities")
    for key, record in payload["items"].items():
        if not isinstance(record, dict) or record.get("item_key") != key:
            raise TransferIntegrityError(f"receipt item identity is malformed: {key}")
        spec = expected[key]
        fields = {**spec.as_dict(), "validator_kind": spec.validator_kind,
                  "validator_value": spec.validator_value}
        fields.pop("validator")
        if any(record.get(k) != v for k, v in fields.items()) or record.get("bytes") != spec.expected_bytes:
            raise TransferIntegrityError(f"receipt record disagrees with approved contract: {key}")
    return payload


def _write_receipt(path: Path, receipt: dict) -> None:
    atomic_write_json(path, receipt)


def _receipt_matches(record: dict, spec: TransferSpec, target: Path) -> bool:
    return (record.get("dataset_id") == spec.dataset_id and record.get("url") == spec.url
            and record.get("destination") == spec.destination and record.get("version") == spec.version
            and record.get("expected_bytes") == spec.expected_bytes and record.get("sha256") == spec.sha256
            and record.get("validator_kind") == spec.validator_kind
            and record.get("validator_value") == spec.validator_value
            and target.is_file() and target.stat().st_size == spec.expected_bytes
            and sha256_file(target) == spec.sha256)


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def _read_sidecar(path: Path, spec: TransferSpec) -> dict:
    try:
        data = _strict_json(path)
    except (OSError, ValueError, TransferApprovalError) as error:
        raise TransferIntegrityError(f"invalid partial-transfer sidecar: {path}") from error
    expected = {"url": spec.url, "version": spec.version, "expected_bytes": spec.expected_bytes,
                "sha256": spec.sha256, "validator_kind": spec.validator_kind,
                "validator_value": spec.validator_value, "schema_version": 1}
    if any(data.get(k) != v for k, v in expected.items()):
        raise TransferIntegrityError(f"partial transfer sidecar does not match approval: {path}")
    return data


def _write_sidecar(path: Path, spec: TransferSpec) -> None:
    atomic_write_json(path, {"schema_version": 1, "url": spec.url, "version": spec.version,
                             "expected_bytes": spec.expected_bytes, "sha256": spec.sha256,
                             "validator_kind": spec.validator_kind, "validator_value": spec.validator_value})


def _content_range(headers) -> tuple[int, int, int]:
    value = _header(headers, "Content-Range")
    if value is None:
        raise TransferIntegrityError("206 response lacks Content-Range")
    match = _CONTENT_RANGE.fullmatch(value)
    if not match:
        raise TransferIntegrityError(f"invalid Content-Range: {value!r}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _transfer_one(spec: TransferSpec, *, root: Path, opener: Callable, chunk: int,
                  socket_timeout: float, wall_timeout: float, max_bytes: int) -> dict:
    target = _target(root, spec.destination)
    part = target.with_name(target.name + ".part")
    meta = _sidecar(part)
    _target(root, str(part.relative_to(root)))
    _target(root, str(meta.relative_to(root)))
    if target.exists():
        raise TransferIntegrityError(f"refusing to overwrite existing destination: {spec.destination}")
    if part.exists() != meta.exists():
        raise TransferIntegrityError(f"partial payload and sidecar must appear together: {part.name}")
    offset = part.stat().st_size if part.exists() else 0
    if offset > spec.expected_bytes:
        raise TransferIntegrityError(f"partial payload exceeds approved size: {part}")
    if part.exists():
        _read_sidecar(meta, spec)

    restart_count = 0
    if offset == spec.expected_bytes:
        if sha256_file(part) != spec.sha256:
            raise TransferIntegrityError(f"complete-sized partial has wrong SHA-256: {part}")
        part.replace(target)
        return {"dataset_id": spec.dataset_id, "destination": spec.destination, "bytes": spec.expected_bytes,
                "sha256": spec.sha256, "restart_count": restart_count, "resumed": True}

    request = urllib.request.Request(spec.url, headers={"User-Agent": "sdd-launch-transfer/1"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
        request.add_header("If-Range", spec.validator_value)
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        # Socket timeout bounds connection/header operations; the body also has
        # an independent wall deadline. This is not a process-wide hard timeout.
        response = _call_opener(opener, request, min(socket_timeout, wall_timeout))
        with _bounded_response(response, started + wall_timeout):
            status = int(getattr(response, "status", 200))
            headers = getattr(response, "headers", None)
            final_url = response.geturl() if callable(getattr(response, "geturl", None)) else None
            if final_url is not None and final_url != spec.url:
                raise TransferIntegrityError(f"response redirected away from approved URL: {final_url}")
            validator = _header(headers, spec.validator_kind.replace("_", "-"))
            if validator != spec.validator_value:
                raise TransferIntegrityError(f"{spec.dataset_id}: response validator changed or is absent")
            if offset and status == 200:
                # A server ignoring Range is safe only if we restart from zero
                # into the same .part file.  We never append a full body to a
                # partial body and never touch the final destination here.
                restart_count = 1
                offset = 0
                mode = "wb"
                declared = _integer_header(headers, "Content-Length")
                if declared != spec.expected_bytes:
                    raise TransferIntegrityError("Range fallback returned the wrong full Content-Length")
            elif offset:
                if status != 206:
                    raise TransferIntegrityError(f"resume requires HTTP 206, got {status}")
                start, end, total = _content_range(headers)
                if start != offset or total != spec.expected_bytes or end != total - 1:
                    raise TransferIntegrityError("Content-Range does not match the requested offset and approved total")
                declared = _integer_header(headers, "Content-Length")
                if declared != end - start + 1:
                    raise TransferIntegrityError("206 Content-Length disagrees with Content-Range")
                mode = "ab"
            else:
                if status != 200:
                    raise TransferIntegrityError(f"initial transfer requires HTTP 200, got {status}")
                declared = _integer_header(headers, "Content-Length")
                if declared != spec.expected_bytes:
                    raise TransferIntegrityError("initial Content-Length disagrees with approved size")
                mode = "wb"
            _write_sidecar(meta, spec)
            received = offset
            deadline = started + wall_timeout
            reader = _TimedReader(response)
            with part.open(mode) as sink:
                try:
                    while True:
                        block = reader.read(chunk, deadline)
                        if not block:
                            break
                        if not isinstance(block, (bytes, bytearray)):
                            raise TransferIntegrityError("response.read() did not return bytes")
                        received += len(block)
                        if received > spec.expected_bytes or received > max_bytes:
                            raise TransferIntegrityError("transfer exceeded approved byte cap")
                        sink.write(block)
                        sink.flush()
                        os.fsync(sink.fileno())
                finally:
                    reader.close()
            if received != spec.expected_bytes:
                raise TransferIntegrityError(f"received {received} of {spec.expected_bytes} approved bytes; partial retained")
    except Exception:
        # The .part and sidecar are intentionally retained for a safe retry.
        raise

    if part.stat().st_size != spec.expected_bytes:
        raise TransferIntegrityError("local partial size disagrees with approved size")
    digest = sha256_file(part)
    if digest != spec.sha256:
        raise TransferIntegrityError(f"SHA-256 mismatch for {spec.dataset_id}; partial retained for diagnosis")
    if target.exists():
        raise TransferIntegrityError(f"destination appeared during transfer: {target}")
    part.replace(target)
    return {"dataset_id": spec.dataset_id, "destination": spec.destination, "bytes": spec.expected_bytes,
            "sha256": digest, "restart_count": restart_count, "resumed": bool(offset)}


def execute_transfers(requested: Iterable[TransferSpec], *, root: str | Path, approval_path: str | Path,
                      receipt_path: str | Path | None = None, opener: Callable | None = None,
                      chunk: int = 1 << 16, socket_timeout: float = 30.0, wall_timeout: float = 900.0,
                      max_bytes: int | None = None, max_total_bytes: int | None = None) -> dict:
    """Execute only an exact approved batch into ``root``.

    No call is made to the network until the approval and every destination
    contract has passed validation.  The caller must opt into this function;
    the launch CLI keeps it behind a separate explicit start flag.
    """
    requested = _validate_requested(requested)
    approval, approved = read_transfer_approval(approval_path)
    _bind_approval(approved, requested)
    if chunk <= 0 or socket_timeout <= 0 or wall_timeout <= 0:
        raise TransferIntegrityError("chunk and timeouts must be positive")
    root = Path(os.path.abspath(root))
    _assert_fresh_root(root)
    receipt_path = Path(os.path.abspath(receipt_path)) if receipt_path else root / "transfer_receipt.json"
    try:
        receipt_relative = str(receipt_path.relative_to(root))
    except ValueError as error:
        raise TransferPathError("transfer receipt must be inside the transfer root") from error
    _target(root, receipt_relative)
    occupied = {receipt_relative, ".sdd-transfer.lock"}
    for spec in requested:
        for relative in (spec.destination, spec.destination + ".part", spec.destination + ".part.meta.json"):
            _target(root, relative)
            if any(relative == other or relative.startswith(other + "/") or other.startswith(relative + "/")
                   for other in occupied):
                raise TransferPathError(f"payload/control path collision: {relative}")
            occupied.add(relative)
    for path in root.rglob("*"):
        if path.is_symlink() or (path.is_file() and str(path.relative_to(root)) not in occupied):
            raise TransferPathError(f"transfer root contains unrelated or symlinked input: {path}")
    receipt = _load_receipt(receipt_path, allow_missing=True)
    if receipt.get("approved_items"):
        existing = [_parse_spec(item, context="receipt contract") for item in receipt["approved_items"]]
        _bind_approval(existing, requested)
        if receipt.get("approval_id") != approval["approval_id"]:
            raise TransferIntegrityError("transfer receipt belongs to a different approval_id")
    receipt.update(approval_id=approval["approval_id"], approved_items=[s.as_dict() for s in requested])
    # Validate every pre-existing destination and partial before any network call.
    for spec in requested:
        target = _target(root, spec.destination)
        previous = receipt["items"].get(_item_key(spec))
        part = _target(root, spec.destination + ".part")
        meta = _target(root, spec.destination + ".part.meta.json")
        if target.exists():
            if previous is not None and _receipt_matches(previous, spec, target):
                continue
            # A matched retained sidecar is the pending publication journal.
            # It survives rename until receipt commit, avoiding a crash window.
            if not target.is_file() or not meta.is_file() or part.exists() \
                    or target.stat().st_size != spec.expected_bytes or sha256_file(target) != spec.sha256:
                raise TransferIntegrityError(f"refusing to overwrite existing destination: {spec.destination}")
            _read_sidecar(meta, spec)
            continue
        if part.exists() != meta.exists():
            raise TransferIntegrityError("partial payload and sidecar must appear together")
        if part.exists():
            if not part.is_file() or part.stat().st_size > spec.expected_bytes:
                raise TransferIntegrityError("partial payload is invalid or exceeds approved size")
            _read_sidecar(meta, spec)
    opener = opener or _open_without_redirects
    cap = max_bytes if max_bytes is not None else max(spec.expected_bytes for spec in requested)
    if cap < max(spec.expected_bytes for spec in requested):
        raise TransferIntegrityError("max_bytes is smaller than an approved payload")
    total_expected = sum(spec.expected_bytes for spec in requested)
    if max_total_bytes is not None and total_expected > max_total_bytes:
        raise TransferIntegrityError("approved batch exceeds the total byte cap")
    records = []
    lock = root / ".sdd-transfer.lock"
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise TransferIntegrityError("transfer root is locked; inspect stale/active lock before retry") from error
    os.close(handle)
    try:
        for spec in requested:
            target = _target(root, spec.destination)
            previous = receipt["items"].get(_item_key(spec))
            if previous is not None and _receipt_matches(previous, spec, target):
                records.append({"dataset_id": spec.dataset_id, "status": "skipped_verified", **previous})
                continue
            if target.exists():
                result = {"dataset_id": spec.dataset_id, "destination": spec.destination,
                          "bytes": spec.expected_bytes, "sha256": spec.sha256,
                          "restart_count": 0, "resumed": True, "recovered_receipt_commit": True}
            else:
                result = _transfer_one(spec, root=root, opener=opener, chunk=chunk,
                                       socket_timeout=socket_timeout, wall_timeout=wall_timeout, max_bytes=cap)
            record = {**result, "item_key": _item_key(spec), "url": spec.url, "version": spec.version,
                      "validator_kind": spec.validator_kind, "validator_value": spec.validator_value,
                      "expected_bytes": spec.expected_bytes,
                      "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            receipt["items"][_item_key(spec)] = record
            _write_receipt(receipt_path, receipt)
            _target(root, spec.destination + ".part.meta.json").unlink(missing_ok=True)
            records.append({"dataset_id": spec.dataset_id, "status": "downloaded", **record})
    finally:
        lock.unlink()
    return {"status": "completed", "approval_id": approval["approval_id"], "items": records,
            "receipt": str(receipt_path)}


def verify_transfer_receipt(root: str | Path, receipt_path: str | Path, *,
                            approval_path: str | Path | None = None,
                            dataset_ids: Iterable[str] | None = None) -> list[str]:
    root = Path(os.path.abspath(root))
    problems = []
    try:
        receipt_path = Path(os.path.abspath(receipt_path))
        _target(root, str(receipt_path.relative_to(root)))
        receipt = _load_receipt(receipt_path)
        specs = [_parse_spec(item, context="receipt contract") for item in receipt["approved_items"]]
        if not receipt["items"] or set(receipt["items"]) != {_item_key(s) for s in specs}:
            raise TransferIntegrityError("receipt is empty or incomplete for its approved batch")
        if approval_path is not None:
            approval, approved = read_transfer_approval(approval_path)
            _bind_approval(approved, specs)
            if receipt.get("approval_id") != approval["approval_id"]:
                raise TransferIntegrityError("receipt approval_id mismatch")
        if dataset_ids and not set(dataset_ids) <= {s.dataset_id for s in specs}:
            raise TransferIntegrityError("receipt does not cover requested dataset ids")
    except (TransferApprovalError, TransferIntegrityError, TransferPathError, ValueError, OSError) as error:
        return [str(error)]
    for dataset_id, record in receipt["items"].items():
        try:
            target = _target(root, record["destination"])
        except (TransferPathError, KeyError) as error:
            problems.append(f"{dataset_id}: {error}")
            continue
        if not target.is_file():
            problems.append(f"{dataset_id}: missing {record['destination']}")
        elif target.stat().st_size != record.get("expected_bytes"):
            problems.append(f"{dataset_id}: size mismatch")
        elif sha256_file(target) != record.get("sha256"):
            problems.append(f"{dataset_id}: SHA-256 mismatch")
    return problems
