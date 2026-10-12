"""Plan-bound execution approval, separate from download approval.

This is an auditable operator acknowledgement, not a cryptographic identity
service. Synthetic exemption requires reproducing the actual generated inputs;
dataset names and a synthetic flag alone cannot authorize a real-data queue.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..common.io import read_json, sha256_file


class ExecutionApprovalError(PermissionError):
    """A reviewed, plan-bound receipt is required before dispatch."""


def generated_synthetic_only(protocol: dict, files: dict, inputs: dict, planned=None) -> bool:
    """Compare bounded diagnostic data against the installed Poisson generator.

    The two-million-element cap bounds the regeneration cost. Larger datasets
    are not exempt: they require ordinary approval, even if labelled synthetic.
    Real H5AD matrices are opened backed and never materialized by this check.
    """
    if protocol.get("kind") != "population" or inputs or not files:
        return False
    reps = protocol.get("representations", {})
    used = {j.representation for j in planned} if planned is not None else set(reps)
    if not used or any(reps.get(name, {}).get("kind") != "pca_train" for name in used):
        return False
    import anndata
    import numpy as np
    from scipy.sparse import csr_matrix
    from ..data.adapters import synthetic_dataset
    from ..data.standard import REQUIRED_OBS

    for dataset_id, info in files.items():
        data = anndata.read_h5ad(info["dataset_file"], backed="r")
        try:
            meta = data.uns.get("standard", {})
            provenance = json.loads(meta.get("provenance_json", "{}"))
            if provenance.get("synthetic") is not True or meta.get("dataset_id") != dataset_id:
                return False
            if data.n_obs * data.n_vars > 2_000_000 or not set(REQUIRED_OBS) <= set(data.obs):
                return False
            sizes = data.obs.groupby(["individual", "time_value"], observed=True).size()
            if sizes.empty or sizes.nunique() != 1 or provenance.get("mode") not in {"translation", "reweighting", "rotation"}:
                return False
            expected = synthetic_dataset(dataset_id=dataset_id, n_individuals=data.obs.individual.nunique(),
                                         times=tuple(data.obs.time_value.unique()), cells=int(sizes.iloc[0]),
                                         genes=data.n_vars, mode=provenance["mode"], seed=provenance["seed"])
            if not np.array_equal(data.var_names, expected.var_names) or data.shape != expected.shape:
                return False
            if any(not np.array_equal(data.obs[c].to_numpy(), expected.obs[c].to_numpy()) for c in REQUIRED_OBS):
                return False
            if (csr_matrix(data.X[:]) != expected.X).nnz:
                return False
        finally:
            data.file.close()
    return True


def discovery_input_paths() -> list:
    from ..measurement import stability, worker
    return [stability.COUNTS, stability.GENES, stability.CELLS, worker.SPECIMENS,
            f"{stability.DISCOVERY}/panel.npy", f"{stability.DISCOVERY}/beta.npy",
            f"{stability.DISCOVERY}/topic_model.pt"]


def validate_execution_scope(protocol, planned, files, provenance, material):
    """Validate consumed input closure, not just mutually consistent hashes."""
    ids = {j.dataset_id for j in planned}
    if not planned or set(material) != ids:
        raise ExecutionApprovalError("review material must cover exactly every planned dataset")
    kind = protocol.get("kind")
    inputs = provenance.get("inputs", {})
    if kind == "population":
        if inputs or set(files) != ids or any(j.job_type != "population_fit" for j in planned):
            raise ExecutionApprovalError("population queue has incompatible job types or input closure")
    elif kind == "measurement":
        if set(inputs) != set(discovery_input_paths()):
            raise ExecutionApprovalError("measurement discovery input closure is incomplete")
        discovery = protocol["cohorts"]["discovery"]
        external = {j.dataset_id for j in planned if j.job_type == "semantics" and j.params.get("source") == "standard"}
        if set(files) != external:
            raise ExecutionApprovalError("measurement manifest must match external semantics datasets")
        for job in planned:
            if job.job_type not in {"semantics", "stability", "design"}:
                raise ExecutionApprovalError("measurement queue has incompatible job type")
            if job.job_type == "semantics" and job.params.get("source") not in {"standard", "discovery"}:
                raise ExecutionApprovalError("unknown semantics input source")
            if (job.job_type != "semantics" or job.params["source"] == "discovery") and job.dataset_id != discovery:
                raise ExecutionApprovalError("discovery job has an inconsistent dataset identity")
    else:
        raise ExecutionApprovalError("unsupported protocol kind")
    for job in planned:
        row = material[job.dataset_id]
        if job.dataset_id in files:
            expected = files[job.dataset_id]["dataset_sha256"]
            if job.params.get("dataset_sha256") != expected or row.get("dataset_sha256") != expected:
                raise ExecutionApprovalError("job, manifest and review material disagree on dataset hash")
            if row.get("basis") != "stage_a" or row.get("contract", {}).get("status") != "accepted":
                raise ExecutionApprovalError("standardized dataset needs an accepted Stage-A contract")
        elif row.get("basis") != "frozen_discovery_inputs" or row.get("input_sha256") != inputs:
            raise ExecutionApprovalError("discovery review does not bind actual frozen inputs")
        if kind == "population":
            if job.task.get("dataset_id") != job.dataset_id or job.representation not in protocol["representations"]:
                raise ExecutionApprovalError("population task or representation identity mismatch")


def approval_binding(run_dir) -> dict:
    """Path-independent identities; receipts survive copying an intact plan."""
    run_dir = Path(run_dir)
    provenance = read_json(run_dir / "provenance.json")
    from .jobs import protocol_hash, read_jobs
    protocol = read_json(run_dir / "protocol.json")
    planned = read_jobs(run_dir / "jobs.jsonl")
    files = read_json(run_dir / "datasets.json")
    return {"schema_version": 1, "purpose": "stage_b_execution",
            "jobs_sha256": sha256_file(run_dir / "jobs.jsonl"),
            "protocol_hash": protocol_hash(protocol), "code_digest": provenance["code_digest"],
            "review_material_sha256": sha256_file(run_dir / "review_material.json"),
            "dataset_ids": sorted({j.dataset_id for j in planned}),
            "dataset_sha256": {k: v["dataset_sha256"] for k, v in sorted(files.items())},
            "input_sha256": provenance.get("inputs", {})}


def approval_request(run_dir) -> dict:
    """Produce an explicitly unapproved review template, never self-authorize."""
    return {**approval_binding(run_dir), "approved": False, "approved_by": "",
            "approved_utc": "", "reviewed_dataset_ids": [],
            "stage_a_reviewed": False, "review_notes": ""}


def require_execution_approval(run_dir, approval_path, *, protocol, files, provenance) -> dict:
    run_dir = Path(run_dir)
    material = run_dir / "review_material.json"
    if not material.exists() or provenance.get("review_material_sha256") != sha256_file(material):
        raise ExecutionApprovalError("review materials changed or are unsealed; create a fresh plan")
    from .jobs import read_jobs
    planned = read_jobs(run_dir / "jobs.jsonl")
    material_data = read_json(material)
    validate_execution_scope(protocol, planned, files, provenance, material_data)
    if provenance.get("execution_scope") == "verified_synthetic":
        if not generated_synthetic_only(protocol, files, provenance.get("inputs", {}), planned):
            raise ExecutionApprovalError("synthetic exemption failed input regeneration")
        return {"scope": "verified_synthetic", "approval_required": False}
    if approval_path is None:
        raise ExecutionApprovalError("Stage B requires --approval FILE after reviewing approval_request.json and review_material.json")
    receipt = read_json(approval_path)
    expected = approval_binding(run_dir)
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ExecutionApprovalError(f"approval does not match the sealed plan: {key}")
    if receipt.get("approved") is not True or receipt.get("stage_a_reviewed") is not True:
        raise ExecutionApprovalError("receipt is an unapproved draft or Stage A was not reviewed")
    if sorted(receipt.get("reviewed_dataset_ids", [])) != expected["dataset_ids"]:
        raise ExecutionApprovalError("review must cover every planned dataset, including discovery inputs")
    for key in ("approved_by", "review_notes"):
        if not isinstance(receipt.get(key), str) or not receipt[key].strip():
            raise ExecutionApprovalError(f"approval lacks {key}")
    try:
        stamp = datetime.fromisoformat(receipt["approved_utc"].replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp > datetime.now(timezone.utc):
            raise ValueError("approval timestamp must have a timezone and not be in the future")
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ExecutionApprovalError("invalid approved_utc timestamp") from error
    for dataset_id, info in material_data.items():
        if info.get("basis") == "frozen_discovery_inputs":
            if not provenance.get("inputs"):
                raise ExecutionApprovalError("discovery review lacks frozen input identities")
        elif info.get("status") != "accepted" or not info.get("probe"):
            raise ExecutionApprovalError(f"{dataset_id}: reviewed Stage-A probe and accepted contract required")
    return {"scope": "approved_real_data", "approval_required": True,
            "receipt_sha256": sha256_file(approval_path), "receipt": receipt}
