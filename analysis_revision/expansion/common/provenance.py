"""Code and input digests that pin a run: a changed file cannot silently mix into completed jobs."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .io import sha256_file

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
CORE_FILES = ("core_engines/data_adapter.py", "core_engines/gse241132.py", "core_engines/topic_simplex_engine.py",
              "core_engines/celltype_markers.py")


def code_files(root=None) -> list:
    root = Path(root) if root is not None else PACKAGE_PARENT
    files = [p for p in (root / "expansion").rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".json"}]
    # The historical explicit list above is retained as a compatibility
    # reference, but a current source closure must not silently omit a newly
    # added engine (for example flow, DPMM or manifold code).  Include every
    # authored Python engine under the package boundary and keep the result
    # deterministic.  This is deliberately narrower than hashing the whole
    # workspace: raw data, manuscripts and generated outputs are not code.
    engine_root = root / "core_engines"
    if engine_root.is_dir():
        files += [p for p in engine_root.rglob("*.py")
                  if p.is_file() and "__pycache__" not in p.parts]
    files += [root / name for name in CORE_FILES if (root / name).exists()]
    return sorted(set(files))


def code_digest(root=None) -> str:
    """Digest of the running code (by default the directory holding expansion/ and core_engines/)."""
    root = Path(root) if root is not None else PACKAGE_PARENT
    digest = hashlib.sha256()
    for path in code_files(root):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()[:20]


def input_digests(root, relative_paths) -> dict:
    root = Path(root)
    return {str(relative): sha256_file(root / relative) for relative in relative_paths}
