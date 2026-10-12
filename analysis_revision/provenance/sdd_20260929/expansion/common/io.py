"""Hashing, canonical JSON and atomic writes."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _default(value: Any):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=_default, allow_nan=False)


def content_hash(value: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def sanitize(value: Any) -> Any:
    """Convert numpy scalars/arrays to Python values and non-finite floats to None."""
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return [sanitize(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _atomic_bytes(path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def atomic_write_text(path, text: str) -> None:
    _atomic_bytes(path, text.encode("utf-8"))


def atomic_write_json(path, value: Any) -> None:
    text = json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False, default=_default, allow_nan=False)
    _atomic_bytes(path, (text + "\n").encode("utf-8"))


def read_json(path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def atomic_save_npz(path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(handle)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
