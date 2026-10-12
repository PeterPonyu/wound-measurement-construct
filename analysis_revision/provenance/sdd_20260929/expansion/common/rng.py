"""Deterministic seed derivation: every random stream is named by its purpose."""
from __future__ import annotations

import hashlib

import numpy as np

from .io import canonical_json


def derive_seed(*parts) -> int:
    digest = hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % (2**31 - 1) + 1


def np_rng(*parts) -> np.random.Generator:
    return np.random.default_rng(derive_seed(*parts))


def seed_torch(seed: int) -> None:
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
