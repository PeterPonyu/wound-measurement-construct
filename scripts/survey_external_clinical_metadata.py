#!/usr/bin/env python3
"""Survey external clinical metadata."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
from datetime import datetime, timezone
import pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_CONTRACT_PATH = os.path.join(ROOT, 'scripts', 'validate_clinical_metadata.py')
_SPEC = importlib.util.spec_from_file_location('dfu_cohort_contract', _CONTRACT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError('cannot load scripts/validate_clinical_metadata.py')
_CONTRACT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONTRACT)
validate_metadata = _CONTRACT.validate_metadata
SCRIPT = os.path.abspath(__file__)
FORBIDDEN_NAME_MARKERS = ('raw.tar', 'counts.csv', 'expression.csv', 'fpkm', 'expressed_gene', '.h5', '.h5ad', 'matrix.mtx', 'barcodes.tsv', 'features.tsv')
SRA_RE = re.compile('(SRX\\d+)')
BIOSAMPLE_RE = re.compile('(SAMN\\d+)')
CONTRACT_FIELDS = ['sample_id', 'patient_id', 'patient_id_source', 'wound_id', 'timepoint_days', 'healing_status', 'follow_up_days', 'tissue_site', 'batch', 'library_id', 'sequencing_run']

def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _clean(value: object) -> str:
    if value is None:
        return ''
    text = str(value).strip().strip('"')
    if text.lower() in {'', 'nan', 'none', 'na', 'null'}:
        return ''
    return text

def _assert_matrix_only(path: str) -> None:
    name = os.path.basename(path).lower()
    if 'series_matrix' not in name:
        raise ValueError(f'refusing to read a non-series-matrix file: {path}')
    if any((marker in name for marker in FORBIDDEN_NAME_MARKERS)):
        raise ValueError(f'refusing to open an expression-like file: {path}')

def parse_series_header(path: str) -> dict[str, list[list[str]]]:
    """Parse GEO series-matrix header tags. Stops before any value table."""
    _assert_matrix_only(path)
    opener = gzip.open if path.endswith('.gz') else open
    tags: dict[str, list[list[str]]] = {}
    with opener(path, 'rt', errors='replace') as handle:
        for line in handle:
            if line.startswith('!series_matrix_table_begin'):
                break
            if not line.startswith('!'):
                continue
            parts = line.rstrip('\n').split('\t')
            key = parts[0]
            values = [_clean(part) for part in parts[1:]]
            tags.setdefault(key, []).append(values)
    return tags

def _first_row(tags: dict[str, list[list[str]]], key: str) -> list[str]:
    rows = tags.get(key) or []
    return rows[0] if rows else []

def _series_text(tags: dict[str, list[list[str]]], key: str) -> str:
    row = _first_row(tags, key)
    return row[0] if row else ''

def _constant_or_blank(values: list[str]) -> str:
    uniq = sorted({_clean(value) for value in values if _clean(value)})
    if len(uniq) == 1:
        return uniq[0]
    return ''

def _characteristics(tags: dict[str, list[list[str]]], n_samples: int) -> list[dict[str, str]]:
    rows = [{} for _ in range(n_samples)]
    for values in tags.get('!Sample_characteristics_ch1', []):
        padded = values + [''] * (n_samples - len(values))
        for index, raw in enumerate(padded[:n_samples]):
            text = _clean(raw)
            if ': ' in text:
                field, value = text.split(': ', 1)
            elif ':' in text:
                field, value = text.split(':', 1)
            else:
                continue
            rows[index][_clean(field).lower()] = _clean(value)
    return rows

def _relation_ids(tags: dict[str, list[list[str]]], pattern: re.Pattern[str], n_samples: int) -> list[str]:
    found = [''] * n_samples
    for values in tags.get('!Sample_relation', []):
        padded = values + [''] * (n_samples - len(values))
        for index, raw in enumerate(padded[:n_samples]):
            match = pattern.search(raw)
            if match and (not found[index]):
                found[index] = match.group(1)
    return found

def _blank_frame(sample_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({'sample_id': sample_ids, 'patient_id': [''] * len(sample_ids), 'patient_id_source': [''] * len(sample_ids), 'wound_id': [''] * len(sample_ids), 'timepoint_days': [''] * len(sample_ids), 'healing_status': [''] * len(sample_ids), 'follow_up_days': [''] * len(sample_ids), 'tissue_site': [''] * len(sample_ids), 'batch': [''] * len(sample_ids), 'library_id': [''] * len(sample_ids), 'sequencing_run': [''] * len(sample_ids)})

def load_common(accession: str) -> tuple[dict[str, list[list[str]]], list[str], list[str], list[str], list[dict[str, str]]]:
    path = os.path.join(ROOT, 'data', 'raw', accession, f'{accession}_series_matrix.txt.gz')
    tags = parse_series_header(path)
    sample_ids = _first_row(tags, '!Sample_geo_accession')
    titles = _first_row(tags, '!Sample_title')
    if not sample_ids:
        raise ValueError(f'{accession}: no GSM accessions in series matrix')
    characteristics = _characteristics(tags, len(sample_ids))
    return (tags, sample_ids, titles, [_clean(title) for title in titles], characteristics)

def build_gse268834() -> tuple[pd.DataFrame, dict]:
    tags, sample_ids, _raw_titles, titles, characteristics = load_common('GSE268834')
    frame = _blank_frame(sample_ids)
    frame['patient_id_source'] = 'geo_title_only'
    frame['tissue_site'] = [row.get('tissue') or 'skin' for row in characteristics]
    frame['library_id'] = _relation_ids(tags, SRA_RE, len(sample_ids))
    frame['batch'] = 'GSE268834_10x_Chromium'
    frame['sequencing_run'] = 'GSE268834_Illumina_NovaSeq_6000'
    notes = {'accession': 'GSE268834', 'assay': 'scRNA-seq', 'n_gsm': len(sample_ids), 'titles': titles, 'disease_labels': [row.get('tissue type') for row in characteristics], 'patient_id_source_reason': 'GEO deposits titles such as SC_W4 and diabetic versus non-diabetic tissue type. That is not an authoritative subject identifier and is not a healing endpoint.', 'batch_source': 'GEO extract protocol: 10x Genomics single-cell suspensions from diabetic and non-diabetic wounds. No per-sample chemistry version.', 'sequencing_run_source': 'GEO !Sample_instrument_model is Illumina NovaSeq 6000 for all eight samples. No flowcell identifiers.', 'independent_prognostic_validation_eligible': False, 'why_not_independent_prognostic_validation': ['no authoritative patient_id', 'no healed/not_healed labels', 'diabetic versus non-diabetic is not a healing endpoint']}
    return (frame, notes)

def build_gse272918() -> tuple[pd.DataFrame, dict]:
    tags, sample_ids, _raw_titles, titles, characteristics = load_common('GSE272918')
    frame = _blank_frame(sample_ids)
    patients = []
    timepoints = []
    for row in characteristics:
        number = _clean(row.get('patient number'))
        patients.append(number)
        point = _clean(row.get('time point')).lower()
        if point == 'before treatment':
            timepoints.append(0)
        elif point == 'after treatment':
            timepoints.append(7)
        else:
            timepoints.append('')
    frame['patient_id'] = patients
    frame['patient_id_source'] = ['authoritative' if patient else '' for patient in patients]
    frame['timepoint_days'] = timepoints
    frame['tissue_site'] = [row.get('tissue') or 'granulation tissue' for row in characteristics]
    frame['library_id'] = _relation_ids(tags, SRA_RE, len(sample_ids))
    frame['batch'] = 'GSE272918_Illumina_TruSeq_stranded_mRNA'
    frame['sequencing_run'] = 'GSE272918_Illumina_HiSeq_2500'
    notes = {'accession': 'GSE272918', 'assay': 'bulk_rnaseq', 'n_gsm': len(sample_ids), 'titles': titles, 'geo_patient_numbers': patients, 'treatments': [row.get('treatment') for row in characteristics], 'patient_id_source_reason': "GEO !Sample_characteristics_ch1 includes depositor field 'patient number' for all six samples. That is authoritative subject identity, not a title parse.", 'timepoint_source': 'GEO time point is before/after treatment; overall design states full-thickness wound-edge tissue before and after 1 week of NPWT. timepoint_days 0 and 7 are that documented interval, not a healing follow-up window.', 'batch_source': 'GEO extract protocol: Illumina TruSeq Stranded mRNA library prep. Series-level process ID only.', 'sequencing_run_source': 'GEO !Sample_instrument_model is Illumina HiSeq 2500 for all six samples. No flowcell identifiers.', 'paper_note': 'PMID 39301661 validated FUS/ILF2 by qPCR in 24 additional DFU patients and reported a 4-week healing association in that qPCR set. Those 24 patients are not in this GEO series. Their healing labels are not assigned to the three RNA-seq patients.', 'independent_prognostic_validation_eligible': False, 'why_not_independent_prognostic_validation': ['bulk RNA-seq cannot enter the frozen scRNA-seq expression representation encoder', 'no healed/not_healed labels for the three sequenced patients', 'no wound_id', 'no follow_up_days for a healing endpoint', 'n=3 is below the registered design-power floor']}
    return (frame, notes)

def build_gse248247() -> tuple[pd.DataFrame, dict]:
    tags, sample_ids, _raw_titles, titles, characteristics = load_common('GSE248247')
    frame = _blank_frame(sample_ids)
    frame['patient_id_source'] = 'geo_title_only'
    frame['tissue_site'] = [row.get('tissue') or 'plantar foot ulcer' for row in characteristics]
    frame['library_id'] = _relation_ids(tags, SRA_RE, len(sample_ids))
    frame['batch'] = 'GSE248247_10x_Chromium_3p_v3'
    frame['sequencing_run'] = 'GSE248247_Illumina_HiSeq_4000'
    notes = {'accession': 'GSE248247', 'assay': 'scRNA-seq', 'n_gsm': len(sample_ids), 'titles': titles, 'patient_id_source_reason': 'Two samples titled human diabetic foot ulcer and human non-diabetic ulcer. The extract protocol mentions one DFU patient but does not name that person. Titles are not promoted to patient IDs.', 'independent_prognostic_validation_eligible': False, 'why_not_independent_prognostic_validation': ['no authoritative patient_id', 'no healed/not_healed labels', 'n=2 and diabetic versus non-diabetic is not a healing endpoint']}
    return (frame, notes)

def build_gse219078() -> tuple[pd.DataFrame, dict]:
    tags, sample_ids, _raw_titles, titles, characteristics = load_common('GSE219078')
    frame = _blank_frame(sample_ids)
    frame['patient_id_source'] = 'geo_title_only'
    frame['tissue_site'] = [row.get('tissue') or 'skin' for row in characteristics]
    frame['library_id'] = _relation_ids(tags, SRA_RE, len(sample_ids))
    frame['batch'] = 'GSE219078_TruSeq_stranded_mRNA_HDMEC_culture'
    frame['sequencing_run'] = 'GSE219078_Illumina_NovaSeq_6000'
    notes = {'accession': 'GSE219078', 'assay': 'bulk_rnaseq_cultured_hdmec', 'n_gsm': len(sample_ids), 'titles': titles, 'disease_state': [row.get('disease state') for row in characteristics], 'patient_id_source_reason': 'Titles are HDMEC control/DFU sample1–4. That is a culture replicate label, not an authoritative patient ID.', 'independent_prognostic_validation_eligible': False, 'why_not_independent_prognostic_validation': ['cultured endothelial cells, not wound single-cell tissue', 'no authoritative patient_id', 'no healed/not_healed labels']}
    return (frame, notes)

def build_gse327462() -> tuple[pd.DataFrame, dict]:
    tags, sample_ids, _raw_titles, titles, characteristics = load_common('GSE327462')
    frame = _blank_frame(sample_ids)
    frame['patient_id_source'] = 'cell_line'
    frame['tissue_site'] = [row.get('cell line') or 'HaCaT cell' for row in characteristics]
    frame['library_id'] = _relation_ids(tags, SRA_RE, len(sample_ids))
    frame['batch'] = 'GSE327462_VAHTS_Universal_V8_RNAseq'
    frame['sequencing_run'] = 'GSE327462_Illumina_NovaSeq_X_Plus'
    notes = {'accession': 'GSE327462', 'assay': 'bulk_rnaseq_cell_line', 'n_gsm': len(sample_ids), 'titles': titles, 'patient_id_source_reason': 'All six samples are HaCaT si_NC / si_YWHAZ replicates. A cell line perturbation is not a patient-linked DFU cohort.', 'independent_prognostic_validation_eligible': False, 'why_not_independent_prognostic_validation': ['HaCaT cell line, not human DFU tissue', 'no patient_id', 'no healing endpoint']}
    return (frame, notes)
BUILDERS = {'GSE268834': build_gse268834, 'GSE272918': build_gse272918, 'GSE248247': build_gse248247, 'GSE219078': build_gse219078, 'GSE327462': build_gse327462}

def _contract_payload(frame: pd.DataFrame, scope: str, protocol: dict) -> dict:
    result = validate_metadata(frame[CONTRACT_FIELDS], scope=scope)
    result['protocol'] = dict(protocol, analysis_scope=scope)
    return result

def survey_one(accession: str, output_dir: str, protocol: dict) -> dict:
    frame, notes = BUILDERS[accession]()
    csv_path = os.path.join(output_dir, f'{accession}_contract_input.csv')
    os.makedirs(output_dir, exist_ok=True)
    frame[CONTRACT_FIELDS].to_csv(csv_path, index=False)
    prognostic = _contract_payload(frame, 'prognostic', protocol)
    projection = _contract_payload(frame, 'projection', protocol)
    return {'accession': accession, 'assay': notes['assay'], 'n_gsm': notes['n_gsm'], 'n_named_patient_rows': int(sum((1 for value in frame['patient_id'] if _clean(value)))), 'n_named_patients': int(pd.Series([value for value in frame['patient_id'] if _clean(value)]).nunique()), 'patient_id_source_values': sorted({_clean(value) for value in frame['patient_id_source'] if _clean(value)}), 'prognostic': {'status': prognostic['status'], 'errors': prognostic['errors'], 'warnings': prognostic['warnings'], 'patient_count': prognostic['patient_count']}, 'projection': {'status': projection['status'], 'errors': projection['errors'], 'warnings': projection['warnings']}, 'independent_prognostic_validation_eligible': notes['independent_prognostic_validation_eligible'], 'why_not_independent_prognostic_validation': notes['why_not_independent_prognostic_validation'], 'notes': notes, 'contract_csv': csv_path, 'contracts': {'prognostic': prognostic, 'projection': projection}}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default=os.path.join(ROOT, 'cohort_metadata', 'next_public_cohorts'))
    parser.add_argument('--report', default=os.path.join(ROOT, 'outputs', 'cohort_metadata', 'next_public_cohort_contract', 'report.json'))
    args = parser.parse_args()
    files_read = []
    for accession in BUILDERS:
        path = os.path.join(ROOT, 'data', 'raw', accession, f'{accession}_series_matrix.txt.gz')
        _assert_matrix_only(path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        files_read.append(os.path.abspath(path))
    protocol = {'script_sha256': sha256(SCRIPT), 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'does_not_open_expression_matrix': True, 'files_read': files_read, 'series_matrix_sha256': {os.path.basename(path): sha256(path) for path in files_read}, 'mcid_filled': False, 'dryad_17_patient_bulk': {'doi': '10.5061/dryad.2v6wwpzzc', 'sra': 'PRJNA1200081', 'status': 'metadata_file_not_locally_available', 'reason': 'Dryad Patients.xlsx is the author sample key for a 17-patient 12-week bulk DFU RNA-seq. The public landing page is readable; the file stream is BotStopper-gated from this environment. Even after a local copy arrives, the assay is bulk and cannot enter frozen expression representation experiment A.'}}
    cohorts = [survey_one(accession, args.output_dir, protocol) for accession in BUILDERS]
    any_prognostic_accepted = any((cohort['prognostic']['status'] == 'accepted' for cohort in cohorts))
    any_independent_prognostic_validation = any((cohort['independent_prognostic_validation_eligible'] for cohort in cohorts))
    gse272918 = next((cohort for cohort in cohorts if cohort['accession'] == 'GSE272918'))
    if gse272918['patient_id_source_values'] != ['authoritative']:
        raise RuntimeError('GSE272918 lost its GEO patient-number authority')
    if gse272918['n_named_patients'] != 3:
        raise RuntimeError('GSE272918 patient count is no longer 3')
    report = {'status': 'survey_complete_no_admitted_cohort', 'independent_prognostic_validation_admitted': False, 'not_an_independent_validation': True, 'any_prognostic_accepted': any_prognostic_accepted, 'mcid_filled': False, 'n_series_surveyed': len(cohorts), 'prognostic_accepted_count': int(sum((cohort['prognostic']['status'] == 'accepted' for cohort in cohorts))), 'projection_accepted_count': int(sum((cohort['projection']['status'] == 'accepted' for cohort in cohorts))), 'closest_public_hit': {'accession': 'GSE272918', 'why_closest': 'Only surveyed series with a GEO-deposited patient number field.', 'why_still_rejected': gse272918['why_not_independent_prognostic_validation']}, 'cohorts': [{key: value for key, value in cohort.items() if key != 'contracts'} for cohort in cohorts], 'full_contracts': {cohort['accession']: cohort['contracts'] for cohort in cohorts}, 'protocol': protocol, 'scientific_scope': 'public metadata-contract survey only; no expression opened; no independent validation; not experiment A'}
    os.makedirs(os.path.dirname(args.report) or '.', exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    summary = {'status': report['status'], 'independent_prognostic_validation_admitted': report['independent_prognostic_validation_admitted'], 'any_prognostic_accepted': any_prognostic_accepted, 'n_series_surveyed': report['n_series_surveyed'], 'cohorts': [{'accession': cohort['accession'], 'assay': cohort['assay'], 'prognostic': cohort['prognostic']['status'], 'projection': cohort['projection']['status'], 'errors': cohort['prognostic']['errors']} for cohort in cohorts], 'report': args.report}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if any_prognostic_accepted or any_independent_prognostic_validation or report['mcid_filled']:
        return 2
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
