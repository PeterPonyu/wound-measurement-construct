#!/usr/bin/env python3
"""Validate clinical metadata."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
import pandas as pd
SCRIPT = os.path.abspath(__file__)
ALLOWED_STATUS = {'healed', 'not_healed'}
IDENTITY_FIELDS = {'sample_id', 'patient_id', 'patient_id_source', 'wound_id', 'timepoint_days'}
PROGNOSTIC_FIELDS = {'healing_status', 'follow_up_days'}
PROVENANCE_FIELDS = {'tissue_site', 'batch', 'library_id', 'sequencing_run'}
CAUSAL_FIELDS = {'treatment', 'age', 'sex', 'bmi', 'hba1c'}

def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _clean(value: object) -> str:
    if value is None:
        return ''
    try:
        if bool(pd.isna(value)):
            return ''
    except (TypeError, ValueError):
        pass
    return str(value).strip()

def _missing(df: pd.DataFrame, field: str) -> int:
    values = df[field].map(_clean)
    return int((values == '').sum())

def _required_fields(scope: str) -> set[str]:
    fields = set(IDENTITY_FIELDS) | set(PROVENANCE_FIELDS)
    if scope in {'prognostic', 'causal'}:
        fields |= PROGNOSTIC_FIELDS
    if scope == 'causal':
        fields |= CAUSAL_FIELDS
    return fields

def validate_metadata(df: pd.DataFrame, *, scope: str) -> dict:
    """Return a structured contract result without modifying ``df``."""
    errors: list[str] = []
    warnings: list[str] = []
    required = _required_fields(scope)
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        errors.append('missing required columns: ' + ', '.join(missing_columns))
        return {'contract_version': '1.0', 'status': 'rejected', 'scope_requested': scope, 'row_count': int(len(df)), 'patient_count': None, 'required_fields': sorted(required), 'missing_columns': missing_columns, 'errors': errors, 'warnings': warnings, 'checks': {}}
    checks: dict[str, object] = {}
    for field in sorted(required):
        n_missing = _missing(df, field)
        checks[f'no_missing_{field}'] = n_missing == 0
        if n_missing:
            errors.append(f'{field} has {n_missing} missing/blank values')
    sample = df['sample_id'].map(_clean)
    patient = df['patient_id'].map(_clean)
    wound = df['wound_id'].map(_clean)
    source = df['patient_id_source'].map(lambda x: _clean(x).lower())
    checks['sample_id_unique'] = bool(sample.is_unique)
    if not sample.is_unique:
        errors.append('sample_id is not unique')
    checks['patient_id_source_authoritative'] = bool((source == 'authoritative').all())
    non_authoritative = sorted(set(source[source != 'authoritative']))
    if non_authoritative:
        errors.append("patient_id_source must be exactly 'authoritative'; found " + ', '.join(non_authoritative))
    proxy_ids = [p for p in patient if p.upper().startswith(('GSM', 'SAMPLE', 'LIB'))]
    checks['patient_ids_do_not_look_like_sample_proxies'] = not proxy_ids
    if proxy_ids:
        errors.append('patient_id contains sample/library-like proxy values')
    tuple_frame = pd.DataFrame({'patient_id': patient, 'wound_id': wound, 'timepoint_days': df['timepoint_days']})
    checks['patient_wound_timepoint_unique'] = not tuple_frame.duplicated().any()
    if tuple_frame.duplicated().any():
        errors.append('duplicate patient_id + wound_id + timepoint_days rows')
    timepoint = pd.to_numeric(df['timepoint_days'], errors='coerce')
    checks['timepoint_days_numeric_nonnegative'] = bool(timepoint.notna().all() and (timepoint >= 0).all())
    if not checks['timepoint_days_numeric_nonnegative']:
        errors.append('timepoint_days must be numeric and non-negative')
    patient_count = int(patient.nunique())
    checks['patient_count'] = patient_count
    if patient_count < 2:
        warnings.append('fewer than two independent patients; no comparative inference')
    if scope in {'prognostic', 'causal'}:
        status = df['healing_status'].map(lambda x: _clean(x).lower())
        invalid_status = sorted(set(status[~status.isin(ALLOWED_STATUS)]))
        checks['healing_status_values_valid'] = not invalid_status
        if invalid_status:
            errors.append('healing_status must use healed/not_healed; found ' + ', '.join(invalid_status))
        present_status = set(status)
        checks['both_healing_classes_present'] = ALLOWED_STATUS <= present_status
        if not checks['both_healing_classes_present']:
            errors.append('both healed and not_healed patients are required')
        status_by_patient = df.assign(_patient=patient, _status=status).groupby('_patient', dropna=False)['_status'].nunique(dropna=False)
        checks['healing_status_consistent_within_patient'] = bool((status_by_patient <= 1).all())
        if not checks['healing_status_consistent_within_patient']:
            errors.append('healing_status conflicts across samples from one patient')
        follow_up = pd.to_numeric(df['follow_up_days'], errors='coerce')
        checks['follow_up_days_numeric_nonnegative'] = bool(follow_up.notna().all() and (follow_up >= 0).all())
        if not checks['follow_up_days_numeric_nonnegative']:
            errors.append('follow_up_days must be numeric and non-negative')
    if scope == 'causal':
        for field in sorted(CAUSAL_FIELDS):
            if field in {'age', 'bmi', 'hba1c'}:
                values = pd.to_numeric(df[field], errors='coerce')
                checks[f'{field}_numeric'] = bool(values.notna().all())
                if not checks[f'{field}_numeric']:
                    errors.append(f'{field} must be numeric for causal scope')
        checks['treatment_has_two_levels'] = df['treatment'].map(_clean).nunique() >= 2
        if not checks['treatment_has_two_levels']:
            errors.append('treatment must contain at least two observed levels')
    baseline_counts = df.assign(_patient=patient, _time=timepoint).query('_time == 0').groupby('_patient').size()
    checks['baseline_row_for_each_patient'] = bool(baseline_counts.reindex(patient.unique(), fill_value=0).gt(0).all())
    if not checks['baseline_row_for_each_patient']:
        warnings.append('one or more patients lack a timepoint_days=0 row')
    status = 'accepted' if not errors else 'rejected'
    return {'contract_version': '1.0', 'status': status, 'scope_requested': scope, 'row_count': int(len(df)), 'patient_count': patient_count, 'required_fields': sorted(required), 'missing_columns': [], 'errors': errors, 'warnings': warnings, 'checks': checks, 'scope_statement': 'accepted for metadata intake only; no expression-level or clinical inference is implied' if status == 'accepted' else 'rejected; keep the cohort descriptive until the listed defects are fixed'}

def _self_test() -> int:
    valid = pd.DataFrame([{'sample_id': 'S1', 'patient_id': 'P1', 'patient_id_source': 'authoritative', 'wound_id': 'W1', 'timepoint_days': 0, 'healing_status': 'healed', 'follow_up_days': 84, 'tissue_site': 'foot', 'batch': 'B1', 'library_id': 'L1', 'sequencing_run': 'R1', 'treatment': 'standard', 'age': 61, 'sex': 'F', 'bmi': 28.0, 'hba1c': 7.2}, {'sample_id': 'S2', 'patient_id': 'P1', 'patient_id_source': 'authoritative', 'wound_id': 'W1', 'timepoint_days': 28, 'healing_status': 'healed', 'follow_up_days': 84, 'tissue_site': 'foot', 'batch': 'B1', 'library_id': 'L2', 'sequencing_run': 'R1', 'treatment': 'standard', 'age': 61, 'sex': 'F', 'bmi': 28.0, 'hba1c': 7.2}, {'sample_id': 'S3', 'patient_id': 'P2', 'patient_id_source': 'authoritative', 'wound_id': 'W2', 'timepoint_days': 0, 'healing_status': 'not_healed', 'follow_up_days': 84, 'tissue_site': 'foot', 'batch': 'B2', 'library_id': 'L3', 'sequencing_run': 'R2', 'treatment': 'standard', 'age': 67, 'sex': 'M', 'bmi': 31.0, 'hba1c': 8.4}, {'sample_id': 'S4', 'patient_id': 'P2', 'patient_id_source': 'authoritative', 'wound_id': 'W2', 'timepoint_days': 28, 'healing_status': 'not_healed', 'follow_up_days': 84, 'tissue_site': 'foot', 'batch': 'B2', 'library_id': 'L4', 'sequencing_run': 'R2', 'treatment': 'standard', 'age': 67, 'sex': 'M', 'bmi': 31.0, 'hba1c': 8.4}])
    accepted = validate_metadata(valid, scope='prognostic')
    assert accepted['status'] == 'accepted', accepted
    rejected = valid.copy()
    rejected.loc[0, 'patient_id_source'] = 'title_proxy'
    rejected.loc[1, 'sample_id'] = 'S1'
    failed = validate_metadata(rejected, scope='prognostic')
    assert failed['status'] == 'rejected', failed
    assert any(('authoritative' in e for e in failed['errors']))
    assert any(('sample_id' in e for e in failed['errors']))
    print(json.dumps({'self_test': 'PASS', 'accepted_fixture': accepted['status'], 'rejected_fixture': failed['status']}, indent=2))
    return 0

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', help='CSV/TSV metadata file from the new cohort')
    parser.add_argument('--delimiter', default=',', help='CSV delimiter (default: comma)')
    parser.add_argument('--scope', choices=('projection', 'prognostic', 'causal'), default='prognostic', help='analysis scope to admit (default: prognostic)')
    parser.add_argument('--output', default='outputs/analysis/next_cohort_contract/report.json')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    if not args.metadata:
        parser.error('--metadata is required unless --self-test is used')
    df = pd.read_csv(args.metadata, sep=args.delimiter, dtype=object)
    result = validate_metadata(df, scope=args.scope)
    result['protocol'] = {'script_sha256': sha256(SCRIPT), 'metadata_path': os.path.abspath(args.metadata), 'analysis_scope': args.scope, 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'does_not_open_expression_matrix': True}
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps({'status': result['status'], 'scope': args.scope, 'rows': result['row_count'], 'patients': result['patient_count'], 'errors': result['errors'], 'warnings': result['warnings'], 'report': args.output}, indent=2, ensure_ascii=False))
    return 0 if result['status'] == 'accepted' else 2
if __name__ == '__main__':
    raise SystemExit(main())
