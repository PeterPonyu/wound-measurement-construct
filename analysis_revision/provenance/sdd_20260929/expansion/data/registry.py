"""Registry of verified datasets (identifiers, structure, roles, adapters, verification)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..common.io import read_json

DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "registry" / "datasets.json"
TIERS = {"W", "G", "M"}
STATUSES = {"local", "to_fetch"}
ADAPTER_STATUSES = {"validated_local", "pending_probe", "probe_only"}
SOURCE_KINDS = {"geo_suppl", "zenodo", "local"}


@dataclass(frozen=True)
class DatasetEntry:
    id: str
    tier: str
    papers: list
    roles: list
    organism: str
    system: str
    design: dict
    source: dict
    local_path: str
    status: str
    adapter: str
    adapter_args: dict
    adapter_status: str
    verification: dict
    license: str
    notes: str = ""
    extra: dict = field(default_factory=dict)


def load_registry(path=DEFAULT_REGISTRY) -> list:
    payload = read_json(path)
    entries = []
    for item in payload["datasets"]:
        known = {k: item[k] for k in DatasetEntry.__dataclass_fields__ if k in item and k != "extra"}
        extra = {k: v for k, v in item.items() if k not in DatasetEntry.__dataclass_fields__}
        entries.append(DatasetEntry(**known, extra=extra))
    return entries


def validate_registry(entries) -> list:
    problems, seen = [], set()
    for entry in entries:
        if entry.id in seen:
            problems.append(f"duplicate dataset id {entry.id}")
        seen.add(entry.id)
        if entry.tier not in TIERS:
            problems.append(f"{entry.id}: tier {entry.tier!r} not in {sorted(TIERS)}")
        if entry.status not in STATUSES:
            problems.append(f"{entry.id}: status {entry.status!r} not in {sorted(STATUSES)}")
        if entry.adapter_status not in ADAPTER_STATUSES:
            problems.append(f"{entry.id}: adapter_status {entry.adapter_status!r} invalid")
        kind = entry.source.get("kind")
        if kind not in SOURCE_KINDS:
            problems.append(f"{entry.id}: source kind {kind!r} invalid")
        if entry.status == "to_fetch":
            files = entry.source.get("files") or []
            if not files:
                problems.append(f"{entry.id}: to_fetch entry lists no files")
            for item in files:
                url = item.get("url") or (entry.source.get("base_url", "") + item.get("name", ""))
                if not url.startswith(("https://", "file://")):
                    problems.append(f"{entry.id}: file {item.get('name')} has no https URL")
        for key in ("method", "date"):
            if not entry.verification.get(key):
                problems.append(f"{entry.id}: verification.{key} missing")
    return problems


def entry_by_id(entries, dataset_id: str):
    for entry in entries:
        if entry.id == dataset_id:
            return entry
    raise KeyError(f"dataset {dataset_id!r} is not registered")
