"""Content-hashed job specifications and planners."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..common.io import atomic_write_text, canonical_json, content_hash


@dataclass(frozen=True)
class Job:
    job_type: str
    dataset_id: str
    method: str
    seed: int
    representation: str
    task: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    protocol_hash: str = ""

    @property
    def job_id(self) -> str:
        return content_hash(asdict(self), length=20)

    def to_dict(self) -> dict:
        return {**asdict(self), "job_id": self.job_id}

    @staticmethod
    def from_dict(data: dict) -> "Job":
        return Job(**{key: value for key, value in data.items() if key != "job_id"})


def protocol_hash(protocol: dict) -> str:
    return content_hash(protocol, length=20)


def method_base(protocol: dict, name: str) -> str:
    """Protocol method entries may name a registered implementation in 'base' (parameter variants)."""
    return protocol["methods"][name].get("base", name)


TASK_LABELS = {"count_curve": "T-K", "source_swap": "T-SRC"}


def plan_population_jobs(protocol: dict, datasets: dict, *, code_digest: str = "") -> list:
    from ..population.infosets import compatible
    from ..population.methods import REGISTRY
    from ..population.tasks import generate_tasks
    from ..population.metrics import evaluation_contract_options

    evaluation = evaluation_contract_options(protocol.get("evaluation", {}))
    digest = protocol_hash(protocol)
    primary = protocol["representation"]["primary"]
    clocks = protocol.get("clock", {}).get("sensitivity", [])
    clock_methods = set(protocol.get("clock", {}).get("sensitivity_methods", []))
    planned = []
    for dataset_id in sorted(datasets):
        info = datasets[dataset_id]
        if info["contract"].get("status") != "accepted":
            continue
        extra = [name for name, config in protocol["representations"].items()
                 if name != primary and dataset_id in config.get("datasets", [])]
        excluded = set(protocol.get("dataset_task_exclusions", {}).get(dataset_id, {}).get("tasks", []))
        params = {"dataset_sha256": info["dataset_sha256"], **({"code_digest": code_digest} if code_digest else {})}
        if evaluation is not None:
            params.update(evaluation_contract=evaluation["name"], evaluation_draw_seed=evaluation["draw_seed"])
        for task in generate_tasks(info["contract"], protocol["tasks"], dataset_id):
            label = TASK_LABELS.get(task.family, task.kind)
            if label in excluded or (label == "T-SRC" and "T-LIO" in excluded):
                continue
            names = [name for name, config in protocol["methods"].items()
                     if task.kind in config.get("tasks", ["T-MT", "T-LIO"])]
            if label in {"T-K", "T-SRC"}:
                allowed = protocol["tasks"][label].get("methods")
                names = [name for name in names if allowed is None or name in allowed]
            representations = [primary] if label in {"T-K", "T-SRC"} else [primary] + extra
            for method in names:
                implementation = REGISTRY[method_base(protocol, method)]
                spec = implementation.spec
                if not compatible(task.kind, spec.variants) or (label == "T-SRC" and not spec.uses_source):
                    continue
                seeds = protocol["seeds"]["stochastic" if spec.stochastic else "deterministic"]
                method_params = dict(params)
                if evaluation is not None:
                    method_params["evaluation_stochastic"] = bool(getattr(implementation, "evaluation_stochastic", False))
                for representation in representations:
                    for seed in seeds:
                        planned.append(Job("population_fit", dataset_id, method, int(seed), representation,
                                           task.to_dict(), dict(method_params), digest))
                if task.family in {"missing_time", "leave_individual"} and method in clock_methods:
                    for clock in clocks:
                        for seed in seeds:
                            planned.append(Job("population_fit", dataset_id, method, int(seed), primary,
                                               task.to_dict(), {**method_params, "clock": clock}, digest))
    identifiers = [job.job_id for job in planned]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate job identifiers in plan")
    return planned


def write_jobs(path, planned) -> None:
    atomic_write_text(path, "".join(canonical_json(job.to_dict()) + "\n" for job in planned))


def read_jobs(path) -> list:
    loaded = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        job = Job.from_dict(data)
        if job.job_id != data["job_id"]:
            raise ValueError(f"job identifier mismatch for {data['job_id']}")
        loaded.append(job)
    return loaded


def plan_measurement_jobs(protocol: dict, cohorts: dict, *, code_digest: str = "") -> list:
    """Semantics jobs per cohort and family, stability jobs per K and seed, one design job."""
    digest = protocol_hash(protocol)
    pinned = {"code_digest": code_digest} if code_digest else {}
    planned = []
    for cohort_id in sorted(cohorts):
        info = cohorts[cohort_id]
        families = protocol["semantics"]["families"]
        if info["source"] == "discovery":
            families = list(families) + [f for f in protocol["semantics"].get("discovery_only_families", []) if f not in families]
        for family in families:
            params = {"family": family, "source": info["source"]}
            if info["source"] != "discovery":
                params.update(dataset_file=info["dataset_file"], dataset_sha256=info["dataset_sha256"])
            planned.append(Job("semantics", cohort_id, family, 0, "cells", {}, {**params, **pinned}, digest))
    budgets = protocol['stability'].get('step_budgets')
    if budgets is not None and (not isinstance(budgets, list) or not budgets or len(budgets) != len(set(budgets))
                               or any(isinstance(b, bool) or not isinstance(b, int) or b < 1 for b in budgets)):
        raise ValueError('step_budgets must be distinct positive integers')
    for budget in budgets if budgets is not None else [None]:
        for k in protocol["stability"]["K"]:
            for seed in protocol["stability"]["seeds"]:
                planned.append(Job("stability", "GSE165816", "topic_simplex", int(seed), "panel", {},
                                   {"K": int(k), **({'steps': budget} if budget is not None else {}), **pinned}, digest))
    if protocol.get("design", {}).get("enabled", True):
        planned.append(Job("design", "GSE165816", "design_calculator", 0, "patients", {}, dict(pinned), digest))
    identifiers = [job.job_id for job in planned]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate job identifiers in measurement plan")
    return planned
