"""Method interface, registry and shared information-set helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..infosets import compatible, resolve_variant


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    variants: tuple
    stochastic: bool
    uses_gpu: bool
    description: str
    uses_source: bool = True


REGISTRY: dict = {}


def register(cls):
    if cls.spec.name in REGISTRY:
        raise ValueError(f"duplicate method {cls.spec.name}")
    REGISTRY[cls.spec.name] = cls
    return cls


def create(name: str, **params):
    try:
        return REGISTRY[name](**params)
    except KeyError as error:
        raise ValueError(f"unknown method {name!r}; known: {sorted(REGISTRY)}") from error


def methods_for(kind: str, names) -> list:
    return [name for name in names if compatible(kind, REGISTRY[name].spec.variants)]


class Method:
    spec: MethodSpec

    def __init__(self, **params):
        self.params = dict(params)
        self.variant = None
        self.inputs = None

    def fit(self, inputs, *, seed: int = 0, device: str = "cpu") -> dict:
        self.variant = resolve_variant(inputs.kind, self.spec.variants)
        self.inputs = inputs
        return dict(self._fit(inputs, seed=seed, device=device) or {})

    def _fit(self, inputs, *, seed: int, device: str) -> dict:
        return {}

    def predict(self, source, *, individual, source_time, target_time, anchor_time=None, seed: int = 0):
        if self.inputs is None:
            raise RuntimeError("fit must be called before predict")
        prediction = self._predict(np.asarray(source, dtype=np.float64), individual=str(individual),
                                   source_time=float(source_time), target_time=float(target_time),
                                   anchor_time=None if anchor_time is None else float(anchor_time), seed=int(seed))
        return np.asarray(prediction, dtype=np.float32)

    def _predict(self, source, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} must implement _predict")


def interpolation_weight(clock: dict, source_time: float, target_time: float, anchor_time: float) -> float:
    if anchor_time is None:
        raise ValueError("this variant needs an anchor time")
    span = clock[anchor_time] - clock[source_time]
    if span <= 0:
        raise ValueError("anchor must be later than the source on the clock")
    return (clock[target_time] - clock[source_time]) / span


def anchor_cells(inputs, individual: str, anchor_time: float) -> np.ndarray:
    key = (individual, float(anchor_time))
    if key not in inputs.train:
        raise KeyError(f"anchor {key} is not in the permitted training data")
    return np.asarray(inputs.train[key], dtype=np.float64)


def population_pairs(inputs, source_time: float, target_time: float) -> list:
    """Training individuals observed at both times: [(individual, source cells, target cells)]."""
    out = []
    for (individual, time), cells in sorted(inputs.train.items()):
        target_key = (individual, float(target_time))
        if np.isclose(time, source_time) and target_key in inputs.train:
            out.append((individual, np.asarray(cells, dtype=np.float64),
                        np.asarray(inputs.train[target_key], dtype=np.float64)))
    if not out:
        raise ValueError(f"no training individual observed at both {source_time} and {target_time}")
    return out
