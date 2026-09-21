#!/usr/bin/env python3
"""Build discovery patient map."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone
import pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from wound_models.data_adapter import parse_series_matrix
_CONTRACT_PATH = os.path.join(ROOT, 'scripts', 'validate_clinical_metadata.py')
_SPEC = importlib.util.spec_from_file_location('dfu_cohort_contract', _CONTRACT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError('cannot load scripts/validate_clinical_metadata.py')
_CONTRACT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONTRACT)
validate_metadata = _CONTRACT.validate_metadata
AUTHOR_TABLE = os.path.join(ROOT, 'cohort_metadata', 'source', '41467_2021_27801_clinical_table.xlsx')
SERIES_MATRIX = os.path.join(ROOT, 'data', 'raw', 'GSE165816', 'GSE165816_series_matrix.txt.gz')
SAMPLE_TOKEN_RE = re.compile('G\\d+[A-Z]?')
TITLE_TOKEN_RE = re.compile('^\\s*(G\\d+[A-Z]?)\\s*:')
BIOSAMPLE_RE = re.compile('(SAMN\\d+)')
HEALING_MAP = {'DFU-H': 'healed', 'DFU-NH': 'not_healed'}
FOLLOW_UP_DAYS = 84
FOLLOW_UP_SOURCE = 'Theocharidis et al. Nat Commun 13:181 Methods: DFU patients followed 12 weeks post-surgery'
EXPECTED_INSTRUMENT = 'Illumina NovaSeq 6000'
BATCH_ID = 'GSE165816_Emory_10x_Chromium_3p_V2V3'
SEQUENCING_RUN_ID = 'GSE165816_Illumina_NovaSeq_S4'
BATCH_SOURCE = "GEO !Sample_extract_protocol_ch1 is identical for all 54 samples: Chromium 3'V2 and 3 reagent kits (10X genomics) processed at Emory University. No per-sample V2 versus V3 assignment is released."
SEQUENCING_RUN_SOURCE = 'GEO !Sample_instrument_model is Illumina NovaSeq 6000 for all 54 samples; the same extract-protocol row states sequencing on the Novaseq S4 instrument. Series note: raw data not available due to patient privacy; !Sample_data_row_count=0. No SRR or flowcell identifier is released.'

def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _clean(value: object) -> str:
    if value is None:
        return ''
    text = str(value).strip()
    if text in {'', '-', 'nan', 'None'}:
        return ''
    return text

def _tokens(value: object) -> list[str]:
    return SAMPLE_TOKEN_RE.findall(_clean(value))

def _title_token(title: object) -> str:
    match = TITLE_TOKEN_RE.match(_clean(title))
    return match.group(1) if match else ''

def _series_constant(values: list[str], field: str) -> str:
    uniq = sorted({_clean(v) for v in values})
    if len(uniq) != 1:
        raise ValueError(f'{field} is not constant across samples: {uniq[:6]}')
    if not uniq[0]:
        raise ValueError(f'{field} is blank')
    return uniq[0]

def parse_series_provenance(path: str) -> pd.DataFrame:
    """Read the GEO tags that parse_series_matrix ignores: instrument, protocol, BioSample."""
    opener = gzip.open if path.endswith('.gz') else open
    gsms: list[str] = []
    instruments: list[str] = []
    protocols: list[list[str]] = []
    relations: list[str] = []
    data_rows: list[str] = []
    with opener(path, 'rt', errors='ignore') as handle:
        for line in handle:
            if line.startswith('!series_matrix_table_begin'):
                break
            if not line.startswith('!Sample_'):
                continue
            parts = [p.strip().strip('"') for p in line.rstrip('\n').split('\t')]
            tag, values = (parts[0], parts[1:])
            if tag == '!Sample_geo_accession':
                gsms = values
            elif tag == '!Sample_instrument_model':
                instruments = values
            elif tag == '!Sample_extract_protocol_ch1':
                protocols.append(values)
            elif tag == '!Sample_relation':
                relations = values
            elif tag == '!Sample_data_row_count':
                data_rows = values
    if not gsms:
        raise ValueError(f'no !Sample_geo_accession in {path}')
    if len(instruments) != len(gsms):
        raise ValueError('instrument_model length does not match GSM count')
    if len(relations) != len(gsms):
        raise ValueError('BioSample relation length does not match GSM count')
    if len(protocols) < 2:
        raise ValueError('expected two !Sample_extract_protocol_ch1 rows')
    instrument = _series_constant(instruments, 'instrument_model')
    if instrument != EXPECTED_INSTRUMENT:
        raise ValueError(f'unexpected instrument_model: {instrument}')
    tissue_protocol = _series_constant(protocols[0], 'tissue extract protocol')
    library_protocol = _series_constant(protocols[1], 'library extract protocol')
    if 'Chromium 3' not in library_protocol:
        raise ValueError("library protocol does not mention Chromium 3' chemistry")
    if 'Novaseq S4' not in library_protocol:
        raise ValueError('library protocol does not mention Novaseq S4')
    if 'Dispase' not in tissue_protocol:
        raise ValueError('tissue protocol does not match the published digestion text')
    row_count = _series_constant(data_rows, 'data_row_count')
    if row_count != '0':
        raise ValueError(f'expected withheld raw reads (data_row_count=0), found {row_count}')
    biosamples = []
    for rel in relations:
        hit = BIOSAMPLE_RE.search(rel)
        if hit is None:
            raise ValueError(f'no SAMN in relation: {rel}')
        biosamples.append(hit.group(1))
    if len(set(biosamples)) != len(gsms):
        raise ValueError('BioSample IDs are not unique per GSM')
    return pd.DataFrame({'gsm': gsms, 'biosample_id': biosamples, 'instrument_model': instrument, 'library_protocol': '10x Chromium 3prime V2 and V3; sequenced on Illumina NovaSeq S4', 'raw_reads_status': 'withheld', 'batch': BATCH_ID, 'sequencing_run': SEQUENCING_RUN_ID, 'batch_source': BATCH_SOURCE, 'sequencing_run_source': SEQUENCING_RUN_SOURCE})

def load_author_table(path: str) -> pd.DataFrame:
    frame = pd.read_excel(path, dtype=object)
    required = {'Group', 'Foot Skin (sample ID in GSE165816)', 'Forearm Biopsy (sample ID in GSE165816)', 'PBMCs (sample ID in GSE165816)'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError('author table missing columns: ' + ', '.join(sorted(missing)))
    if len(frame) != 27:
        raise ValueError(f'expected 27 subject rows, found {len(frame)}')
    return frame

def load_series(path: str) -> pd.DataFrame:
    pheno = parse_series_matrix(path)
    provenance = parse_series_provenance(path)
    merged = pheno.merge(provenance, on='gsm', how='left', validate='one_to_one')
    if merged['biosample_id'].map(_clean).eq('').any():
        raise ValueError('BioSample join left blank rows')
    return merged

def build_map(author: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    series = series.copy()
    series['sample_title'] = series['title'].map(_title_token)
    if series['sample_title'].eq('').any():
        bad = series.loc[series['sample_title'].eq(''), 'title'].tolist()
        raise ValueError('series titles without G-number prefix: ' + ', '.join(bad[:6]))
    if not series['sample_title'].is_unique:
        raise ValueError('duplicate G-number titles in the series matrix')
    series_by_title = series.set_index('sample_title', drop=False)
    rows: list[dict[str, object]] = []
    used: set[str] = set()
    for index, record in author.iterrows():
        patient_id = f'NCOMMS2022_S{int(index) + 1:02d}'
        group = _clean(record['Group'])
        sites = (('foot', record['Foot Skin (sample ID in GSE165816)']), ('forearm', record['Forearm Biopsy (sample ID in GSE165816)']), ('pbmc', record['PBMCs (sample ID in GSE165816)']))
        for tissue_site, raw in sites:
            for token in _tokens(raw):
                if token in used:
                    raise ValueError(f'{token} mapped to more than one subject')
                if token not in series_by_title.index:
                    raise ValueError(f'{token} is absent from the series matrix')
                used.add(token)
                geo = series_by_title.loc[token]
                rows.append({'sample_id': _clean(geo['gsm']), 'sample_title': token, 'geo_title': _clean(geo['title']), 'patient_id': patient_id, 'patient_id_source': 'authoritative', 'patient_id_source_record': f'Nature Communications 13:181 supplementary clinical table row {int(index) + 1}', 'group': group, 'wound_id': 'index_plantar_dfu' if group.startswith('DFU') else 'none', 'timepoint_days': 0, 'healing_status': HEALING_MAP.get(group, ''), 'follow_up_days': FOLLOW_UP_DAYS if group.startswith('DFU') else '', 'follow_up_days_source': FOLLOW_UP_SOURCE if group.startswith('DFU') else '', 'tissue_site': tissue_site, 'geo_tissue': _clean(geo.get('tissue', '')), 'geo_disease': _clean(geo.get('disease', '')), 'biosample_id': _clean(geo['biosample_id']), 'instrument_model': _clean(geo['instrument_model']), 'library_protocol': _clean(geo['library_protocol']), 'raw_reads_status': _clean(geo['raw_reads_status']), 'batch': _clean(geo['batch']), 'library_id': _clean(geo['gsm']), 'sequencing_run': _clean(geo['sequencing_run']), 'batch_source': _clean(geo['batch_source']), 'sequencing_run_source': _clean(geo['sequencing_run_source']), 'age': _clean(record.get('Age (years)')), 'sex': _clean(record.get('Sex')), 'race': _clean(record.get('Race (White / African American)')), 'bmi': _clean(record.get('BMI (kg/m²)')), 'hba1c': _clean(record.get('Hemoglobin A1c (%)')), 'wound_area_cm2': _clean(record.get('Wound Surface Area (cm²)')), 'dm_duration_years': _clean(record.get('DM Duration (years)'))})
    mapped = pd.DataFrame(rows)
    missing_titles = sorted(set(series['sample_title']) - used)
    if missing_titles:
        raise ValueError('series titles not claimed by the author table: ' + ', '.join(missing_titles))
    if len(mapped) != 54:
        raise ValueError(f'expected 54 mapped samples, found {len(mapped)}')
    return mapped

def prognostic_subset(mapped: pd.DataFrame) -> pd.DataFrame:
    foot_dfu = mapped[(mapped['tissue_site'] == 'foot') & mapped['group'].isin(HEALING_MAP)].copy()
    if len(foot_dfu) != 14:
        raise ValueError(f'expected 14 DFU foot samples, found {len(foot_dfu)}')
    patients = foot_dfu['patient_id'].nunique()
    if patients != 11:
        raise ValueError(f'expected 11 DFU patients, found {patients}')
    return foot_dfu

def collapsed_pairs(foot_dfu: pd.DataFrame) -> list[dict[str, object]]:
    pairs = []
    for patient_id, block in foot_dfu.groupby('patient_id', sort=True):
        if len(block) < 2:
            continue
        pairs.append({'patient_id': patient_id, 'group': block['group'].iloc[0], 'sample_titles': block['sample_title'].tolist(), 'sample_ids': block['sample_id'].tolist()})
    return pairs

def collapse_to_patient(foot_dfu: pd.DataFrame) -> pd.DataFrame:
    """One row per patient+wound+timepoint. Replicate GSMs stay in a side column."""
    rows: list[dict[str, object]] = []
    for patient_id, block in foot_dfu.groupby('patient_id', sort=True):
        block = block.sort_values(['sample_title', 'sample_id'])
        first = block.iloc[0]
        rows.append({'sample_id': f'{patient_id}_foot', 'patient_id': patient_id, 'patient_id_source': 'authoritative', 'wound_id': first['wound_id'], 'timepoint_days': first['timepoint_days'], 'healing_status': first['healing_status'], 'follow_up_days': first['follow_up_days'], 'tissue_site': 'foot', 'biosample_id': '|'.join(block['biosample_id'].tolist()), 'instrument_model': first['instrument_model'], 'library_protocol': first['library_protocol'], 'raw_reads_status': first['raw_reads_status'], 'batch': first['batch'], 'library_id': '|'.join(block['sample_id'].tolist()), 'sequencing_run': first['sequencing_run'], 'batch_source': first['batch_source'], 'sequencing_run_source': first['sequencing_run_source'], 'replicate_sample_ids': '|'.join(block['sample_id'].tolist()), 'replicate_sample_titles': '|'.join(block['sample_title'].tolist())})
    return pd.DataFrame(rows)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--author-table', default=AUTHOR_TABLE)
    parser.add_argument('--series-matrix', default=SERIES_MATRIX)
    parser.add_argument('--output-dir', default=os.path.join(ROOT, 'cohort_metadata'))
    parser.add_argument('--contract-output', default=os.path.join(ROOT, 'outputs', 'cohort_metadata', 'gse165816_author_map_contract.json'))
    args = parser.parse_args()
    author = load_author_table(args.author_table)
    series = load_series(args.series_matrix)
    mapped = build_map(author, series)
    foot_dfu = prognostic_subset(mapped)
    sample_input = foot_dfu[['sample_id', 'patient_id', 'patient_id_source', 'wound_id', 'timepoint_days', 'healing_status', 'follow_up_days', 'tissue_site', 'batch', 'library_id', 'sequencing_run']].copy()
    patient_input = collapse_to_patient(foot_dfu)
    contract_fields = ['sample_id', 'patient_id', 'patient_id_source', 'wound_id', 'timepoint_days', 'healing_status', 'follow_up_days', 'tissue_site', 'batch', 'library_id', 'sequencing_run']
    sample_contract = validate_metadata(sample_input, scope='prognostic')
    patient_contract = validate_metadata(patient_input[contract_fields], scope='prognostic')
    protocol = {'script_sha256': sha256(os.path.abspath(__file__)), 'author_table_sha256': sha256(args.author_table), 'series_matrix_sha256': sha256(args.series_matrix), 'analysis_scope': 'prognostic', 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'does_not_open_expression_matrix': True, 'not_an_independent_cohort': True, 'follow_up_days_source': FOLLOW_UP_SOURCE, 'finest_public_provenance': {'batch': BATCH_ID, 'sequencing_run': SEQUENCING_RUN_ID, 'instrument_model': EXPECTED_INSTRUMENT, 'per_sample_chemistry_released': False, 'per_sample_flowcell_ids_released': False, 'raw_reads_status': 'withheld', 'unique_biosamples': int(mapped['biosample_id'].nunique()), 'batch_source': BATCH_SOURCE, 'sequencing_run_source': SEQUENCING_RUN_SOURCE}}
    sample_contract['protocol'] = dict(protocol, unit='sample_with_replicate_rows')
    patient_contract['protocol'] = dict(protocol, unit='collapsed_patient_wound_timepoint')
    contract = {'sample_level': sample_contract, 'patient_collapsed': patient_contract, 'status': patient_contract['status']}
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.contract_output) or '.', exist_ok=True)
    mapped_path = os.path.join(args.output_dir, 'GSE165816_ncomms_subject_sample_map.csv')
    prognostic_path = os.path.join(args.output_dir, 'GSE165816_prognostic_contract_input.csv')
    collapsed_path = os.path.join(args.output_dir, 'GSE165816_prognostic_patient_collapsed.csv')
    mapped.to_csv(mapped_path, index=False)
    sample_input.to_csv(prognostic_path, index=False)
    patient_input.to_csv(collapsed_path, index=False)
    with open(args.contract_output, 'w', encoding='utf-8') as handle:
        json.dump(contract, handle, indent=2, ensure_ascii=False)
    summary = {'subjects_in_author_table': int(len(author)), 'mapped_samples': int(len(mapped)), 'dfu_foot_samples': int(len(foot_dfu)), 'dfu_patients': int(foot_dfu['patient_id'].nunique()), 'healed_patients': int(foot_dfu.loc[foot_dfu['healing_status'] == 'healed', 'patient_id'].nunique()), 'not_healed_patients': int(foot_dfu.loc[foot_dfu['healing_status'] == 'not_healed', 'patient_id'].nunique()), 'collapsed_same_patient_foot_pairs': collapsed_pairs(foot_dfu), 'finest_public_provenance': protocol['finest_public_provenance'], 'sample_level_contract': {'status': sample_contract['status'], 'errors': sample_contract['errors'], 'warnings': sample_contract['warnings']}, 'patient_collapsed_contract': {'status': patient_contract['status'], 'errors': patient_contract['errors'], 'warnings': patient_contract['warnings']}, 'map_csv': mapped_path, 'prognostic_csv': prognostic_path, 'patient_collapsed_csv': collapsed_path, 'contract_report': args.contract_output, 'scientific_scope': 'authoritative discovery-cohort unit map only; not experiment A and not a patient-level confirmatory result'}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if contract['status'] == 'accepted' else 2
if __name__ == '__main__':
    raise SystemExit(main())
