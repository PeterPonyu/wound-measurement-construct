"""Task specifications, clocks and permitted-data views.

``build_task_data`` copies only permitted cells into ``TaskData.train``;
evaluation targets are stored separately and never reach a method.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..common.rng import np_rng
from .infosets import allowed


@dataclass(frozen=True)
class Pair:
    individual: str
    source_time: float
    target_time: float
    anchor_time: float | None = None


@dataclass(frozen=True)
class TaskSpec:
    dataset_id: str
    kind: str
    family: str
    held_time: float | None
    held_individual: str | None
    pairs: tuple
    training_individuals: tuple | None = None
    repeat: int = 0
    substitute: str | None = None

    def __post_init__(self):
        allowed(self.kind)
        if self.kind == "T-MT" and self.held_time is None:
            raise ValueError("T-MT tasks need held_time")
        if self.kind == "T-LIO" and self.held_individual is None:
            raise ValueError("T-LIO tasks need held_individual")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["pairs"] = [dict(pair) for pair in data["pairs"]]
        data["training_individuals"] = (list(self.training_individuals)
                                        if self.training_individuals is not None else None)
        return data

    @staticmethod
    def from_dict(data: dict) -> "TaskSpec":
        subset = data.get("training_individuals")
        return TaskSpec(dataset_id=data["dataset_id"], kind=data["kind"], family=data["family"],
                        held_time=data.get("held_time"), held_individual=data.get("held_individual"),
                        pairs=tuple(Pair(**pair) for pair in data["pairs"]),
                        training_individuals=tuple(subset) if subset is not None else None,
                        repeat=int(data.get("repeat", 0)), substitute=data.get("substitute"))

    @property
    def label(self) -> str:
        held = f"time={self.held_time:g}" if self.kind == "T-MT" else f"individual={self.held_individual}"
        size = "" if self.training_individuals is None else f":k={len(self.training_individuals)}:r={self.repeat}"
        swap = "" if self.substitute is None else f":substitute={self.substitute}"
        return f"{self.dataset_id}:{self.kind}:{self.family}:{held}{size}{swap}"


def make_clock(times, kind: str = "rank") -> dict:
    ordered = sorted({float(t) for t in times})
    if kind == "rank":
        span = max(len(ordered) - 1, 1)
        return {t: index / span for index, t in enumerate(ordered)}
    if kind == "log_days":
        if ordered[0] < 0:
            raise ValueError("log_days needs nonnegative times")
        top = float(np.log1p(ordered[-1]))
        return {t: (float(np.log1p(t)) / top if top > 0 else 0.0) for t in ordered}
    raise ValueError(f"unknown clock {kind!r}")


def select_individuals(individuals, limit, *parts) -> list:
    ordered = sorted(individuals)
    if not limit or len(ordered) <= limit:
        return ordered
    return sorted(np_rng(*parts).choice(ordered, size=int(limit), replace=False).tolist())


def generate_tasks(contract: dict, config: dict, dataset_id: str) -> list:
    generated = []
    if config.get("T-MT", {}).get("enabled", True) and contract["eligible"]["T-MT"]:
        for item in contract["missing_time_tasks"]:
            pairs = tuple(Pair(i, float(item["source_time"]), float(item["held_time"]), float(item["anchor_time"]))
                          for i in item["individuals"])
            generated.append(TaskSpec(dataset_id, "T-MT", "missing_time", float(item["held_time"]), None, pairs))
    lio = config.get("T-LIO", {})
    minimum = int(lio.get("min_individuals", 3))
    if lio.get("enabled", True) and contract["eligible"]["T-LIO"]:
        for individual in select_individuals(contract["lio_individuals"], lio.get("max_held_individuals"),
                                             dataset_id, "T-LIO"):
            pairs = tuple(Pair(individual, float(p["source_time"]), float(p["target_time"]))
                          for p in contract["consecutive_pairs"]
                          if individual in p["individuals"] and len(p["individuals"]) >= minimum)
            if pairs:
                generated.append(TaskSpec(dataset_id, "T-LIO", "leave_individual", None, individual, pairs))
    swap = config.get("T-SRC", {})
    if swap.get("enabled", False) and contract["eligible"]["T-LIO"]:
        # Leave-two-out: held individual i and substitute j are both excluded from fitting, so the own and the
        # substituted source are equally out of sample.
        for individual in select_individuals(contract["lio_individuals"], lio.get("max_held_individuals"),
                                             dataset_id, "T-LIO"):
            peers = [i for i in contract["lio_individuals"] if i != individual]
            for substitute in select_individuals(peers, swap.get("max_substitutes", 3), dataset_id, "T-SRC", individual):
                pairs = tuple(Pair(individual, float(p["source_time"]), float(p["target_time"]))
                              for p in contract["consecutive_pairs"]
                              if individual in p["individuals"] and substitute in p["individuals"]
                              and len(p["individuals"]) >= max(minimum, 3))
                if pairs:
                    generated.append(TaskSpec(dataset_id, "T-LIO", "source_swap", None, individual, pairs,
                                              substitute=substitute))
    curve = config.get("T-K", {})
    pool = contract["lio_individuals"]
    if curve.get("enabled", True) and len(pool) >= int(curve.get("min_individuals", 6)):
        for individual in select_individuals(pool, curve.get("max_held_individuals", 6), dataset_id, "T-K"):
            others = sorted(i for i in pool if i != individual)
            for repeat in range(int(curve.get("repeats", 3))):
                # Nested subsets: one ordering per (held individual, repeat); a larger k extends a smaller k.
                ordering = np_rng(dataset_id, "T-K", individual, repeat).permutation(others).tolist()
                for k in curve.get("k_values", [1, 2, 3, 4, 5, 6, 8, 12, 16]):
                    if k > len(others):
                        continue
                    subset = tuple(sorted(ordering[: int(k)]))
                    pairs = tuple(Pair(individual, float(p["source_time"]), float(p["target_time"]))
                                  for p in contract["consecutive_pairs"]
                                  if individual in p["individuals"] and any(s in p["individuals"] for s in subset))
                    if pairs:
                        generated.append(TaskSpec(dataset_id, "T-LIO", "count_curve", None, individual, pairs,
                                                  subset, repeat))
    return generated


def representation_mask(obs: pd.DataFrame, task: TaskSpec) -> np.ndarray:
    """Cells that may shape the representation: never the held time or held individual."""
    if task.kind == "T-MT":
        return ~np.isclose(obs["time_value"].to_numpy(dtype=float), float(task.held_time))
    excluded = {str(task.held_individual)} | ({str(task.substitute)} if task.substitute is not None else set())
    return (~obs["individual"].astype(str).isin(excluded)).to_numpy()


@dataclass
class TaskData:
    spec: TaskSpec
    train: dict
    sources: dict
    targets: dict
    clock: dict


@dataclass
class MethodInputs:
    kind: str
    train: dict
    clock: dict
    held_time: float | None
    held_individual: str | None


def build_task_data(coords: np.ndarray, obs: pd.DataFrame, task: TaskSpec, clock: dict, *,
                    min_cells: int = 1) -> TaskData:
    individuals = obs["individual"].astype(str).to_numpy()
    times = obs["time_value"].to_numpy(dtype=float)
    groups = {(str(i), float(t)): np.asarray(index)
              for (i, t), index in pd.Series(np.arange(len(obs))).groupby([individuals, times]).indices.items()}
    subset = set(task.training_individuals) if task.training_individuals is not None else None
    train = {}
    for (individual, time), index in groups.items():
        if task.kind == "T-MT":
            permitted = not np.isclose(time, float(task.held_time))
        else:
            permitted = (individual != task.held_individual and individual != task.substitute
                         and (subset is None or individual in subset))
        if permitted and len(index) >= min_cells:
            train[(individual, time)] = coords[index]
    sources, targets = {}, {}
    for pair in task.pairs:
        source_key = (pair.individual, float(pair.source_time))
        target_key = (pair.individual, float(pair.target_time))
        sources[source_key] = coords[groups[source_key]]
        targets[target_key] = coords[groups[target_key]]
        if task.substitute is not None:
            substitute_key = (task.substitute, float(pair.source_time))
            sources[substitute_key] = coords[groups[substitute_key]]
    return TaskData(task, train, sources, targets, clock)


def method_inputs(data: TaskData) -> MethodInputs:
    return MethodInputs(data.spec.kind, data.train, data.clock, data.spec.held_time, data.spec.held_individual)
