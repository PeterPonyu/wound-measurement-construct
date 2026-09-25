"""Failing consistency checks for the authoritative native LaTeX manuscripts."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPTS = ROOT / 'manuscripts'
LEDGER_PATH = MANUSCRIPTS / 'claim_ledger.json'
COVERAGE_PATH = MANUSCRIPTS / 'result_coverage.json'
PAPER_FILES = {'measurement_construct': MANUSCRIPTS / 'latex/manuscript.tex'}
ANCHORS = {'measurement_construct': ['76,461', '243', '0.933', '0.841', '24.9%', '0.0293', '0.272727', '−0.0204', '561', '358', '0.909', '0.948', '0.714', '0.786', '0.937', '+0.0476', '0.0107', '0.0402', '0.2991', '0.166', '1.66', '+0.210', '11.23/15', '0.0015', '5.0%']}
AFFIRMATIVE_PATTERNS = [re.compile('(?i)\\b(?:is|was|constitutes|establishes|supports)\\b[^.\\n]{0,60}\\bvalidated\\s+(?:healing|prognostic|DFU)\\s+biomarker\\b'), re.compile('(?i)\\bTopic[- ]Simplex\\b[^.\\n]{0,45}\\b(?:is|was)\\s+(?:clearly\\s+)?superior\\b'), re.compile('(?i)\\bconditioning\\b[^.\\n]{0,35}\\b(?:is|was)\\s+the\\s+correct\\s+model\\s+class\\b'), re.compile('(?i)\\bdonor\\s+(?:number|count)\\b[^.\\n]{0,35}\\b(?:is|was|remains)\\s+(?:the|a|the\\s+main|the\\s+binding)\\s+bottleneck\\b'), re.compile('(?i)\\bGSE248247\\b[^.\\n]{0,45}\\b(?:validates|confirms|establishes)\\b[^.\\n]{0,35}\\b(?:healing|prognosis|prognostic)\\b'), re.compile('(?i)\\b(?:demonstrates|establishes|proves|validates|enables)\\b[^.\\n]{0,35}\\b(?:DFU|diabetic[- ]foot[- ]ulcer)\\b[^.\\n]{0,20}\\btrajectory\\s+prediction\\b')]

def fail(message: str) -> None:
    print(f'FAIL  {message}')
    raise SystemExit(1)

def read_native(path: Path, stack: tuple[Path, ...]=()) -> str:
    """Read every included scientific section, caption and numerical table."""
    path = path.resolve()
    if path in stack:
        fail(f'cyclic LaTeX include: {path.name}')
    text = path.read_text(encoding='utf-8')

    def include(match: re.Match) -> str:
        child = path.parent / match[1]
        if not child.suffix:
            child = child.with_suffix('.tex')
        return read_native(child, stack + (path,))
    text = re.sub('\\\\input\\{([^}]+)\\}', include, text)
    return text.replace('\\%', '%').replace('{[}', '[').replace('{]}', ']')

def check_author_style() -> None:
    """Check the author's prose, equation and cross-reference requirements."""
    totals = {'equations': 0, 'tables': 0, 'panels': 0}
    for n in (1,):
        folder = MANUSCRIPTS / 'latex'
        main = (folder / 'manuscript.tex').read_text()
        protected = {'abstract': main.split('\\section*{Abstract}', 1)[1].split('\\section{Introduction}', 1)[0], 'introduction': main.split('\\section{Introduction}', 1)[1].split('\\input{methods}', 1)[0], 'discussion': (folder / 'discussion.tex').read_text()}
        abstract = protected['abstract'].split('\\par\\smallskip', 1)[0].replace('\\noindent', '')
        word_count = len(abstract.split())
        if word_count > 250:
            fail(f'Paper {n}: abstract has {word_count} words, limit 250')
        if '\\bibliographystyle{unsrtnat}' not in main:
            fail(f'Paper {n}: bibliography must follow first citation order')
        title = re.search('\\\\papertitle\\{([^}]+)\\}', main)
        metadata = re.search('\\\\hypersetup\\{pdftitle=\\{([^}]+)\\}\\}', main)
        if title is None or metadata is None or title[1] != metadata[1]:
            fail(f'Paper {n}: displayed title differs from PDF metadata')
        cite_command = re.compile('\\\\(?:[Cc]ite[A-Za-z]*|nocite)\\b')
        paths = [folder / 'manuscript.tex'] + [folder / f'{section}.tex' for section in ('methods', 'results', 'discussion', 'supplement', 'tables')] + list((folder / 'figure_captions').glob('*.tex'))
        for path in paths:
            source = re.sub('(?<!\\\\)%[^\\n]*', '', path.read_text())
            if path.name == 'manuscript.tex':
                intro = re.sub('(?<!\\\\)%[^\\n]*', '', protected['introduction'])
                forbidden_citations = source.replace(intro, '')
            elif path.name == 'discussion.tex':
                forbidden_citations = ''
            else:
                forbidden_citations = source
            if cite_command.search(forbidden_citations):
                fail(f'Paper {n}: literature citation outside Introduction / Discussion: {path.name}')
            if not path.name.endswith(('methods.tex', 'results.tex')) and '\\subsection' in source:
                fail(f'Paper {n}: subsections outside Materials and methods / Results: {path.name}')
            if path.name in ('tables.tex','layout.tex'):
                continue
            if re.search('\\\\(?:textbf|textit|emph|paragraph|subsubsection)\\b|\\\\begin\\{(?:itemize|enumerate|description)\\}', source):
                fail(f'Paper {n}: forbidden prose formatting in {path.name}')
        for heading in ('Funding.', 'Competing interests.', 'Author contributions.', 'Ethics approval and consent to participate.', 'Consent for publication.', 'Data availability.', 'Code availability.', 'Acknowledgements.'):
            if heading not in main:
                fail(f'Paper {n}: missing declaration: {heading}')
        for name, text in protected.items():
            if re.search('\\\\(?:ref|eqref|autoref|cref|Cref|pageref|nameref|hyperref|input|includegraphics)\\b|\\b(?:Figure|Fig\\.?|Table|Section|Subsection|Equation)s?\\s*(?:~|\\(?S?\\d)|(?:see|in|above|below)\\s+(?:the\\s+)?Results\\b', text, re.I):
                fail(f'Paper {n} {name}: internal result/figure/table reference')
            if re.search('\\\\(?:textbf|textit|emph|textcolor|colorbox|fcolorbox)\\b|\\\\begin\\{(?:itemize|enumerate|description)\\}', text):
                fail(f'Paper {n} {name}: decorative formatting in narrative prose')
            if re.search('\\b(?:supplementary|supplemental|appendix|appendices)\\b|\\$(?!\\$)', text, re.I):
                fail(f'Paper {n} {name}: supplementary pointer or mathematical styling in plain narrative')
        for section in ('methods', 'supplement'):
            text = (folder / f'{section}.tex').read_text()
            for label in re.finditer('\\\\label\\{(eq:[^}]+)\\}', text):
                tail = text[label.end():]
                end = re.search('\\\\end\\{(?:equation|align)\\}', tail)
                if end is None or '\\eqref{' + label[1] + '}' not in tail[end.end():end.end() + 950]:
                    fail(f'Paper {n}: equation has no adjacent explanatory citation: {label[1]}')
                totals['equations'] += 1
        results_text = (folder / 'results.tex').read_text()
        for section in ('methods', 'supplement'):
            source = (folder / f'{section}.tex').read_text()
            for label in re.findall('\\\\label\\{(eq:[^}]+)\\}', source):
                if '\\eqref{' + label + '}' not in results_text:
                    fail(f'Paper {n}: equation lacks a Results interpretation: {label}')
        panel_counts = {1: [6, 3, 3, 3, 6, 6, 2, 2, 3], 2: [6, 3, 6, 1, 2, 2, 2]}[n]
        for number, count in enumerate(panel_counts, 1):
            key = f'fig:p{n}f{number}'
            expected = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:count]) if count > 1 else set()
            used = set()
            references = list(re.finditer('\\\\ref\\{' + re.escape(key) + '\\}([A-Z])?(?:--([A-Z]))?', results_text))
            if not references:
                fail(f'Paper {n}: figure has no Results citation: {key}')
            for reference in references:
                if count == 1:
                    if reference[1]:
                        fail(f'Paper {n}: single-panel figure has a panel letter: {key}')
                    continue
                if not reference[1]:
                    fail(f'Paper {n}: figure reference lacks a panel letter: {key}')
                used.update((chr(i) for i in range(ord(reference[1]), ord(reference[2] or reference[1]) + 1)))
            caption = (folder / f'figure_captions/f{number}.tex').read_text()
            described = set(re.findall('\\(([A-Z])\\)', caption))
            if used != expected or described != expected:
                fail(f'Paper {n} figure {number}: panel references {used}, caption descriptions {described}, expected {expected}')
            totals['panels'] += len(expected)
        tables = (folder / 'tables.tex').read_text() + (folder / 'supplement.tex').read_text()
        for label in re.findall('\\\\label\\{(tab:[^}]+)\\}', tables):
            if '\\ref{' + label + '}' not in results_text:
                fail(f'Paper {n}: table lacks a Results citation: {label}')
            totals['tables'] += 1
        print(f'PASS  paper {n}: abstract={word_count} words; literature only in Introduction/Discussion; first-citation bibliography; complete declaration categories')
    print(f"PASS  author style: protected prose sections=3, adjacent equation citations={totals['equations']}, main-text table citations={totals['tables']}, individually described/cited panels={totals['panels']}")

def check_revision_claims(ledger: dict) -> None:
    """Check dated sources, derived numbers and the limits of revised claims.

    These are consistency contracts, not an assessment of scientific readiness.
    A one-paper export can retain only its own paper without reading its sibling.
    """
    papers = ledger['papers']
    for n, key in ((1, 'measurement_construct'),):
        if key not in papers:
            continue
        manuscript = (MANUSCRIPTS / 'latex/manuscript.tex').read_text()
        title = re.search('\\\\papertitle\\{([^}]+)\\}', manuscript)
        if title is None or papers[key]['title'] != title[1]:
            fail(f'Paper {n}: claim-ledger title differs from native manuscript')
    claims = {}
    for paper in papers.values():
        for claim in paper['claims']:
            if claim['id'] in claims:
                fail(f"duplicate claim ID: {claim['id']}")
            claims[claim['id']] = claim
    checked = set()
    revision = 'outputs/scientific_revision_20260922/'

    def report(source: str):
        return json.loads((ROOT / source).read_text())

    def require(condition: bool, message: str) -> None:
        if not condition:
            fail(message)

    def check_claim(claim_id: str, source: str, expected: str, wording: tuple[str, ...]=()) -> None:
        claim = claims.get(claim_id, {})
        if claim.get('source') != source or claim.get('value') != expected:
            fail(f'{claim_id}: claim-ledger value/source differs from revised report')
        if any((part not in claim.get('allowed_wording', '') for part in wording)):
            fail(f'{claim_id}: claim-ledger interpretation lost its scope limitation')
        checked.add(claim_id)

    def digest_matches(path: Path, expected: str, claim_id: str) -> None:
        """Authenticate native bytes, or a documented standalone translation.

        Never accept a requested hash, missing source or unverified fallback.
        Exported text may contain path translations: authenticate both the
        exported bytes and the archived native bytes through EXPORT_PROVENANCE.
        """
        require(bool(re.fullmatch('[0-9a-f]{64}', expected)), f'{claim_id}: invalid source hash')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual == expected:
            return
        provenance_path = ROOT / 'EXPORT_PROVENANCE.json'
        if provenance_path.is_file():
            relative = str(path.relative_to(ROOT))
            records = [row for row in json.loads(provenance_path.read_text())['files'] if row.get('exported') == relative]
            if len(records) == 1:
                row = records[0]
                original = ROOT / row.get('original', '__missing_original__')
                if row.get('source_sha256') == expected and row.get('exported_sha256') == actual and original.is_file() and (hashlib.sha256(original.read_bytes()).hexdigest() == expected):
                    return
        fail(f'{claim_id}: source/artifact hash mismatch: {path.name}')

    def current_source(claim_id: str, source: str, manifest_name: str, artifact_names: tuple[str, ...]):
        claim = claims.get(claim_id, {})
        require(claim.get('source') == source, f'{claim_id}: wrong completed source')
        digest_matches(ROOT / source, claim.get('source_sha256', ''), claim_id)
        folder = (ROOT / source).parent
        evidence = claim.get('evidence_sha256', {})
        require(manifest_name in evidence, f'{claim_id}: missing anchored artifact manifest')
        for name, digest in evidence.items():
            digest_matches(folder / name, digest, claim_id)
        manifest = json.loads((folder / manifest_name).read_text())
        if manifest_name == 'artifact_manifest.json':
            manifest = manifest['files_sha256']
        for name in artifact_names:
            require(name in manifest, f'{claim_id}: missing artifact hash {name}')
            digest_matches(folder / name, manifest[name], claim_id)
        return (report(source), folder)

    def rows_csv(folder: Path, name: str):
        return list(csv.DictReader((folder / name).read_text().splitlines()))

    def close(actual, expected, message):
        require(math.isfinite(float(actual)) and math.isfinite(float(expected)) and math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-11), message)

    def keyed(rows, columns, message):
        result = {tuple((str(row[column]) for column in columns)): row for row in rows}
        require(len(result) == len(rows), message + ': duplicate row')
        return result

    def check_full_depth() -> None:
        import itertools
        claim_id = 'P1-FULL-DEPTH'
        source = 'outputs/scientific_revision_20260925/full_pipeline_depth/full/report.json'
        names = ('protocol_frozen.json', 'raw_input_manifest.json', 'baseline_reproduction.json', 'qc_gate_transitions_by_specimen.csv', 'specimen_readouts.csv', 'patient_readouts.csv', 'patient_exact_contrasts.csv', 'conditional_stability_by_run.csv')
        depth, folder = current_source(claim_id, source, 'artifact_manifest.json', names)
        frozen = json.loads((folder / 'protocol_frozen.json').read_text())
        protocol = frozen['protocol']
        baseline = depth['baseline_reproduction']
        contract = json.loads((folder / 'run_contract.json').read_text())
        digest_matches(folder / 'protocol_frozen.json', depth['protocol_frozen_sha256'], claim_id)
        require(depth['status'] == contract['status'] == 'completed' and depth['mode'] == frozen['mode'] == 'full' and (depth['all_input_and_code_hashes_unchanged'] is True) and (contract['model_training'] is False), f'{claim_id}: full-run/no-refit scope changed')
        require(protocol['perturbation']['retention_rates'] == [0.75, 0.5, 0.25, 0.1] and protocol['perturbation']['seeds'] == [20260925, 20260926, 20260927] and (protocol['perturbation']['baseline_rate'] == 1.0) and (protocol['qc'] == {'minimum_detected_genes': 200, 'maximum_mito_fraction': 0.2}), f'{claim_id}: rate/seed/QC protocol changed')
        require((baseline['raw_cells'], baseline['qc_cells'], baseline['fibroblasts'], baseline['specimens']) == (81602, 76461, 19410, 25) and baseline['patients_checked'] == 20 and (baseline['all_cell_patient_aggregation_reproduced'] is True) and (baseline['maximum_coordinate_absolute_difference'] <= 1e-06), f'{claim_id}: raw-versus-retained baseline reproduction failed')
        raw = json.loads((folder / 'raw_input_manifest.json').read_text())
        require(raw['not_the_old_qc_cache'] is True and raw['inputs'] == frozen['inputs'] and (len(raw['samples']) == 25) and (sum((r['evaluated_raw_cells'] for r in raw['samples'])) == 81602), f'{claim_id}: raw/source hash manifest differs from frozen inputs')
        for info in frozen['inputs'].values():
            require(bool(re.fullmatch('[0-9a-f]{64}', info['sha256'])) and info['bytes'] > 0, f'{claim_id}: invalid frozen input hash')
        gsms = {r['gsm'] for r in raw['samples']}
        require(len(gsms) == 25, f'{claim_id}: duplicate raw specimen')
        grid = {(1.0, -1)} | {(rate, seed) for rate in (0.75, 0.5, 0.25, 0.1) for seed in (20260925, 20260926, 20260927)}
        transitions = rows_csv(folder, 'qc_gate_transitions_by_specimen.csv')
        require(len(transitions) == 325 and {(float(r['rate']), int(r['seed']), r['gsm']) for r in transitions} == {(rate, seed, gsm) for rate, seed in grid for gsm in gsms}, f'{claim_id}: incomplete raw sample/rate/seed grid')
        for row in transitions:
            require(int(row['original_qc']) == int(row['common_qc']) + int(row['lost_qc']) and int(row['current_qc']) == int(row['common_qc']) + int(row['gained_qc']) and (int(row['common_qc_lineage_changed']) <= int(row['common_qc'])), f'{claim_id}: QC loss/gain/gate arithmetic differs')
        pooled = {(float(r['rate']), int(r['seed'])): r for r in depth['pooled_qc_gate_by_run']}
        require(set(pooled) == grid and len(depth['pooled_qc_gate_by_run']) == 13, f'{claim_id}: report omitted a run')
        fields = ('n_raw', 'original_qc', 'current_qc', 'common_qc', 'lost_qc', 'gained_qc', 'original_fibroblasts', 'original_fibroblasts_lost_qc', 'current_fibroblasts', 'common_qc_lineage_changed', 'common_qc_fibroblast_exit', 'common_qc_fibroblast_entry', 'current_qc_zero_marker_cells', 'current_qc_exact_marker_ties', 'current_qc_missing_score')
        for pair, summary in pooled.items():
            rows = [r for r in transitions if (float(r['rate']), int(r['seed'])) == pair]
            for field in fields:
                require(summary[field] == sum((int(r[field]) for r in rows)), f'{claim_id}: CSV/report {field} differs')
            require(summary['n_raw'] == 81602 and summary['original_qc'] == 76461, f'{claim_id}: pre-QC cells were omitted')
            close(summary['original_qc_retention_fraction'], summary['common_qc'] / 76461, f'{claim_id}: original retention denominator differs')
            close(summary['common_qc_lineage_change_fraction'], summary['common_qc_lineage_changed'] / summary['common_qc'], f'{claim_id}: gate-change denominator differs')
        populations = {'full_pipeline', 'original_qc_fixed_gate', 'qc_intersection_fixed_gate', 'qc_intersection_regated'}
        readouts = ('all_cell_topic0_mean', 'fibroblast_topic0_mean', 'fibroblast_high_state_fraction', 'fibroblast_fraction')
        patients = rows_csv(folder, 'patient_readouts.csv')
        specimens = rows_csv(folder, 'specimen_readouts.csv')
        require(len(specimens) == 13 * 4 * 25 and len(patients) == 13 * 4 * 20, f'{claim_id}: missing population/sample/patient readouts')
        keyed(specimens, ('run', 'population', 'gsm'), claim_id)
        keyed(patients, ('run', 'population', 'patient_id'), claim_id)
        for patient in patients:
            rows = [r for r in specimens if (r['run'], r['population'], r['patient_id']) == (patient['run'], patient['population'], patient['patient_id'])]
            require(rows and patient['population'] in populations and ({r['arm'] for r in rows} == {patient['arm']}), f'{claim_id}: population or patient identity drift')
            for field in ('n_all', 'n_fibroblast', 'n_missing_score', 'n_missing_fibroblast_score'):
                require(int(patient[field]) == sum((int(r[field]) for r in rows)), f'{claim_id}: patient count weights differ')
            totals = {'all_cell_topic0_mean': ('topic0_sum', 'n_all', 'n_missing_score'), 'fibroblast_topic0_mean': ('fibroblast_topic0_sum', 'n_fibroblast', 'n_missing_fibroblast_score'), 'fibroblast_high_state_fraction': ('fibroblast_high_count', 'n_fibroblast', 'n_missing_fibroblast_score')}
            for field, (numerator, denominator, missing) in totals.items():
                if int(patient[missing]) or int(patient[denominator]) == 0:
                    require(patient[field] == '', f'{claim_id}: missing scores silently dropped')
                else:
                    close(patient[field], sum((float(r[numerator]) for r in rows)) / int(patient[denominator]), f'{claim_id}: all-cell/fibroblast patient aggregation differs')
            close(patient['fibroblast_fraction'], int(patient['n_fibroblast']) / int(patient['n_all']), f'{claim_id}: fibroblast composition denominator differs')
        contrasts = rows_csv(folder, 'patient_exact_contrasts.csv')
        require(len(contrasts) == 13 * 4 * 4, f'{claim_id}: missing outcome contrast rows')
        keyed(contrasts, ('run', 'population', 'readout'), claim_id)
        for row in contrasts:
            eligible = [r for r in patients if r['run'] == row['run'] and r['population'] == row['population'] and (r['arm'] in ('DFU-healer', 'DFU-nonhealer'))]
            require(row['readout'] in readouts and len(eligible) == 11 and (sum((r['arm'] == 'DFU-healer' for r in eligible)) == 7), f'{claim_id}: outcome unit must be 11 patients, not cells/seeds/healthy controls')
            if any((r[row['readout']] == '' for r in eligible)):
                require(row['status'] == 'not_estimable' and row['difference'] == row['exact_p'] == '', f'{claim_id}: missing-patient contrast fabricated')
                continue
            values = [float(r[row['readout']]) for r in eligible]
            observed = sum((v for v, r in zip(values, eligible) if r['arm'] == 'DFU-healer')) / 7 - sum((v for v, r in zip(values, eligible) if r['arm'] == 'DFU-nonhealer')) / 4
            null = [sum((values[i] for i in ids)) / 7 - (sum(values) - sum((values[i] for i in ids))) / 4 for ids in itertools.combinations(range(11), 7)]
            extreme = sum((abs(v) >= abs(observed) - 1e-12 for v in null))
            close(row['difference'], observed, f'{claim_id}: patient contrast differs from patient CSV')
            require(int(row['allocations']) == 330 and int(row['extreme_allocations']) == extreme, f'{claim_id}: exact allocation counts differ')
            close(row['exact_p'], extreme / 330, f'{claim_id}: exact probability differs')
        ranges = {(float(r['rate']), r['readout']): r for r in depth['full_pipeline_outcome_seed_ranges']}
        require(len(ranges) == 20, f'{claim_id}: missing full-pipeline seed ranges')
        for (rate, field), summary in ranges.items():
            rows = [r for r in contrasts if float(r['rate']) == rate and r['population'] == 'full_pipeline' and (r['readout'] == field)]
            require(summary['n_runs'] == summary['n_estimable'] == len(rows), f'{claim_id}: seed count mismatch')
            for metric, column in (('difference', 'difference'), ('exact_p', 'exact_p')):
                close(summary[metric + '_min'], min((float(r[column]) for r in rows)), f'{claim_id}: report min differs')
                close(summary[metric + '_max'], max((float(r[column]) for r in rows)), f'{claim_id}: report max differs')
        tenth = [r for (rate, _), r in pooled.items() if rate == 0.1]
        allcell = ranges[0.1, 'all_cell_topic0_mean']
        check_claim(claim_id, source, f"81602 pre-QC cells; baseline 76461 retained / 19410 fibroblasts; 25 specimens; 4 rates x 3 seeds; 10% depth retains {min((r['current_qc'] for r in tenth))}-{max((r['current_qc'] for r in tenth))} cells; common-QC lineage changes {min((r['common_qc_lineage_change_fraction'] for r in tenth)) * 100:.4f}-{max((r['common_qc_lineage_change_fraction'] for r in tenth)) * 100:.4f}%; 11-patient all-cell difference {allcell['difference_min']:.8f}-{allcell['difference_max']:.8f}; exact p {allcell['exact_p_min']:.6f}-{allcell['exact_p_max']:.6f}; zero-marker forced labels baseline {pooled[1.0, -1]['current_qc_zero_marker_cells']}, 10% {min((r['current_qc_zero_marker_cells'] for r in tenth))}-{max((r['current_qc_zero_marker_cells'] for r in tenth))}", ('not biological replicates', 'not annotation truth or external portability', 'RT06 remains partially addressed', 'not a clinical acceptance margin'))

    def check_paired_clock() -> None:
        claim_id = 'P2-PAIRED-CLOCK'
        source = 'outputs/scientific_revision_20260925/paired_clock/run/paired_clock/report.json'
        clock, folder = current_source(claim_id, source, 'output_manifest.json', ('run_contract.json', 'per_donor_seed.csv', 'summary.csv'))
        contract = json.loads((folder / 'run_contract.json').read_text())
        require(clock['status'] == 'completed' and clock['fits'] == 12 and (clock['biological_donors'] == 3) and (clock['independent_new_data'] is False) and all((clock[k] is True for k in ('paired_random_streams_verified', 'exact_target_time', 'holdout_excluded_from_scaling_and_fitting'))), f'{claim_id}: fit/donor/holdout scope changed')
        axes, donors, seeds = (('rank', 'days', 'sqrt_days', 'log_days'), ('PWH26', 'PWH27', 'PWH28'), (0, 1, 2))
        require(contract['seeds'] == list(seeds) and contract['holdout'] == 'Wound7' and (contract['steps'] == 8000) and (contract['batch_size'] == 256) and (contract['rk4_steps'] == 50) and (contract['evaluation_cap'] == 1200) and (contract['target_used_for_model_selection'] is False) and (set(contract['axes']) == set(axes)), f'{claim_id}: fixed clock training/evaluation protocol changed')
        require(len(contract['inputs']) >= 2 and all((re.fullmatch('[0-9a-f]{64}', v) for v in contract['inputs'].values())), f'{claim_id}: missing input source hashes')
        records = rows_csv(folder, 'per_donor_seed.csv')
        require(len(records) == 36 and {(r['axis'], int(r['seed']), r['donor']) for r in records} == {(axis, seed, donor) for axis in axes for seed in seeds for donor in donors}, f'{claim_id}: incomplete clock/seed/donor grid')
        manifest = json.loads((folder / 'output_manifest.json').read_text())
        for seed in seeds:
            fingerprints = []
            for axis in axes:
                name = f'seed{seed}/{axis}/fit.json'
                digest_matches(folder / name, manifest[name], claim_id)
                fit = json.loads((folder / name).read_text())
                require(fit['axis'] == axis and fit['seed'] == seed and (set(fit['artifacts']) == {*[d + '.npz' for d in donors], 'field.pt'}), f'{claim_id}: fit identity/artifact grid differs')
                fingerprints.append(tuple((fit['random_streams'][k] for k in ('endpoint_draws_sha256', 'initial_weights_sha256', 'initial_interpolation_rng_sha256'))))
                for artifact, digest in fit['artifacts'].items():
                    require(manifest[f'seed{seed}/{axis}/{artifact}'] == digest, f'{claim_id}: fit/source artifact hashes disagree')
            require(len(set(fingerprints)) == 1, f'{claim_id}: random streams not paired within seed')
        summaries = keyed(clock['summary'], ('axis',), claim_id)
        csv_summaries = keyed(rows_csv(folder, 'summary.csv'), ('axis',), claim_id)
        require(set(summaries) == set(csv_summaries) == {(a,) for a in axes}, f'{claim_id}: summary axis coverage changed')
        for row in records:
            require(row['source'] == 'Wound1' and row['target'] == 'Wound7', f'{claim_id}: wrong observed target/source')
            close(row['target_time'], contract['axes'][row['axis']]['Wound7'], f'{claim_id}: not exact target time')
        for axis in axes:
            selected = [r for r in records if r['axis'] == axis]
            for metric in ('flow_ed', 'refined_flow_ed', 'unchanged_source_ed', 'centroid_ed'):
                value = sum((float(r[metric]) for r in selected)) / 9
                close(summaries[axis,][metric], value, f'{claim_id}: donor/seed mean differs from CSV')
                close(csv_summaries[axis,][metric], value, f'{claim_id}: summary CSV differs from raw grid')
        values = '/'.join((f"{summaries[a,]['flow_ed']:.6f}" for a in axes))
        centroid = '/'.join((f"{summaries[a,]['centroid_ed']:.6f}" for a in axes))
        check_claim(claim_id, source, f"12 paired exact-time fits; 3 donors; seeds 0/1/2; 8000 steps; rank/days/sqrt_days/log_days flow ED {values}; centroid ED {centroid}; unchanged ED {summaries['rank',]['unchanged_source_ed']:.6f}", ('historical nearest-grid scan is separate', 'not biological confidence intervals', 'no target-guided best clock', 'not clinical validation'))
        old_source = 'outputs/analysis/time_parameterisation/report.json'
        old = report(old_source)
        require(claims['P2-C04'].get('current_use') == 'historical_comparison_only' and claims['P2-C04'].get('superseded_by') == claim_id, 'P2-C04: historical clock boundary missing')
        check_claim('P2-C04', old_source, f"historical single-seed nearest-grid linear-days improvement {old['axes_results']['days']['improvement_at_claimed'] * 100:.1f}%; retained only in tab:p2axes; current Figure 4 uses P2-PAIRED-CLOCK", ('historical comparison only', 'not the current Figure 4', 'not paired with the revised exact-time fits'))

    def check_observable() -> None:
        claim_id = 'P2-OBSERVABLE'
        source = 'outputs/scientific_revision_20260925/population_observable/final-01/report.json'
        names = ('protocol.json', 'preparation_manifest.json', 'input_sha256.json', 'features.csv', 'panel_coverage.csv', 'split_verification.json', 'observed_expression_per_fit.csv', 'configuration_means.csv', 'donor_blocks.csv', 'macro_scores.csv', 'macro_method_contrasts.csv', 'decoder_reconstruction_reference.csv', 'training_signatures.json')
        observable, folder = current_source(claim_id, source, 'output_manifest.json', names)
        protocol = json.loads((folder / 'protocol.json').read_text())
        digest_matches(folder / 'protocol.json', observable['protocol_sha256'], claim_id)
        digest_matches(folder / 'preparation_manifest.json', observable['preparation_manifest_sha256'], claim_id)
        require(observable['status'] == 'completed' and observable['predictive_scores_started_after_protocol_feature_freeze'] is True and ((observable['biological_donors'], observable['new_biological_donors'], observable['saved_fits'], observable['trained_models'], observable['observed_count_cells'], observable['common_genes'], observable['panel_genes'], observable['scores']) == (3, 0, 27, 0, 12259, 5383, 7002, 324)) and (observable['primary_metric'] == 'total_variation') and (observable['primary_target_set'] == 'original_evaluation') and (observable['primary_target_kind'] == 'actual raw observed count-derived proportions, not decoded latent truth'), f'{claim_id}: observed-target/donor/gene/fit unit changed')
        require(protocol['expected']['seeds'] == [0, 1, 2] and protocol['expected']['donors'] == ['PWH26', 'PWH27', 'PWH28'] and (protocol['source_condition'] == 'Wound1') and (protocol['target_condition'] == 'Wound7'), f'{claim_id}: fixed protocol changed')
        checks = observable['verification']
        require(checks['status'] == 'passed' and all((checks[k] is True for k in ('all_input_hashes_unchanged', 'complete_fit_grid', 'raw_barcode_author_label_and_latent_alignment', 'training_scalers_recomputed', 'saved_inverse_transform_verified', 'same_split_baselines', 'training_only_signatures', 'no_cell_as_biological_replicate'))), f'{claim_id}: input/alignment/exclusion checks failed')
        inputs = json.loads((folder / 'input_sha256.json').read_text())
        require(len(inputs) > 27 and all((re.fullmatch('[0-9a-f]{64}', digest) for digest in inputs.values())), f'{claim_id}: input/source hashes incomplete')
        features, panel = (rows_csv(folder, 'features.csv'), rows_csv(folder, 'panel_coverage.csv'))
        require(len(features) == len({r['gene'] for r in features}) == 5383 and len(panel) == 7002 and ({r['gene'] for r in features} == {r['gene'] for r in panel if r['observed_in_all_archives'] == 'True'}), f'{claim_id}: common-panel features were filtered or missing genes scored as zeros')
        donors, seeds = (('PWH26', 'PWH27', 'PWH28'), (0, 1, 2))
        fits = json.loads((folder / 'split_verification.json').read_text())
        require(len(fits) == 27 and len({r['path'] for r in fits}) == 27, f'{claim_id}: missing fitted-state grid')
        expected_grid = {(held, train, seed) for held in donors for train in [tuple([d]) for d in donors if d != held] + [tuple((d for d in donors if d != held))] for seed in seeds}
        require({(r['held_donor'], tuple(r['training_donors']), r['seed']) for r in fits} == expected_grid and all((r['held_donor'] not in r['training_donors'] and r['inverse_max_abs_error'] <= 1e-12 and (r['mean_displacement_z_max_abs_error'] <= 1e-06) for r in fits)), f'{claim_id}: donor exclusion/seed grid/inverse-transform contract changed')
        methods = ('shared_cfm_decoded', 'persistence_decoded', 'mean_displacement_decoded', 'training_target_marginal_decoded_exact', 'persistence_observed', 'training_target_marginal_observed_exact')
        targets = ('original_evaluation', 'all_day7')
        metrics = ('total_variation', 'js_divergence_nats', 'gene_rmse')
        rows = rows_csv(folder, 'observed_expression_per_fit.csv')
        require(len(rows) == 324 and {(r['fit_path'], r['method'], r['target_set']) for r in rows} == {(f['path'], m, t) for f in fits for m in methods for t in targets}, f'{claim_id}: incomplete fit/method/target observed-score grid')
        for row in rows:
            fit = next((f for f in fits if f['path'] == row['fit_path']))
            require(row['held_donor'] == fit['held_donor'] and int(row['seed']) == fit['seed'] and (row['training_donors'] == '|'.join(fit['training_donors'])) and (int(row['k']) == len(fit['training_donors'])) and (int(row['n_common_genes']) == 5383) and (row['target_kind'] == 'actual_observed_expression'), f'{claim_id}: prediction/observed row identity differs')

        def aggregate(records, columns):
            grouped = {}
            for row in records:
                key = tuple((str(row[c]) for c in columns))
                grouped.setdefault(key, []).append(row)
            return {key: {metric: sum((float(r[metric]) for r in group)) / len(group) for metric in metrics} for key, group in grouped.items()}
        configs = rows_csv(folder, 'configuration_means.csv')
        blocks = rows_csv(folder, 'donor_blocks.csv')
        macros = rows_csv(folder, 'macro_scores.csv')
        for source_rows, target_rows, columns, expected_n in ((rows, configs, ('held_donor', 'k', 'training_donors', 'target_set', 'method'), 108), (configs, blocks, ('held_donor', 'k', 'target_set', 'method'), 72), (blocks, macros, ('k', 'target_set', 'method'), 24)):
            computed = aggregate(source_rows, columns)
            actual = keyed(target_rows, columns, claim_id)
            require(len(actual) == expected_n and set(actual) == set(computed), f'{claim_id}: incomplete equal-donor aggregation')
            for key, vals in computed.items():
                for metric in metrics:
                    close(actual[key][metric], vals[metric], f'{claim_id}: CSV equal seed/subset/donor summary differs')
        reported = keyed(observable['macro'], ('k', 'target_set', 'method'), claim_id)
        actual = keyed(macros, ('k', 'target_set', 'method'), claim_id)
        require(set(reported) == set(actual), f'{claim_id}: report macro grid missing')
        for key, row in reported.items():
            require(row['biological_donors'] == 3 and row['biological_ci_available'] is False, f'{claim_id}: seeds promoted to biological uncertainty')
            for metric in metrics:
                close(row[metric], actual[key][metric], f'{claim_id}: report macro differs from CSV')
        refs = rows_csv(folder, 'decoder_reconstruction_reference.csv')
        require(len(refs) == len(observable['decoder_reference']) == 6, f'{claim_id}: reconstruction reference missing')
        refs_map = keyed(refs, ('held_donor', 'target_set'), claim_id)
        for row in observable['decoder_reference']:
            require(row['method'] == 'decoder_reconstruction_reference_not_forecast', f'{claim_id}: decoder reference promoted to forecast')
            for metric in metrics:
                close(row[metric], refs_map[row['held_donor'], row['target_set']][metric], f'{claim_id}: decoder reference differs')
        primary = {(k, m): reported[str(k), 'original_evaluation', m]['total_variation'] for k in (1, 2) for m in methods}

        def pair(method):
            return f'{primary[1, method]:.6f}/{primary[2, method]:.6f}'
        check_claim(claim_id, source, f"27 saved fits; 3 healthy donors; 12259 observed cells; 5383/7002 genes; no new training; observed-expression TV k1/k2: flow {pair('shared_cfm_decoded')}; mean displacement {pair('mean_displacement_decoded')}; decoded target marginal {pair('training_target_marginal_decoded_exact')}; observed target marginal {pair('training_target_marginal_observed_exact')}; decoded persistence {pair('persistence_decoded')}; observed persistence {pair('persistence_observed')}", ('not absolute counts', 'not a mathematical lower error bound', 'not biological confidence intervals', 'POP05 remains partially addressed', 'not clinical or external validation'))

    def check_measurement() -> None:
        base = revision + 'measurement/'
        patient_source = base + 'patient_unit_remap/report.json'
        patient = report(patient_source)
        effect = patient['effect_cell_weighted']
        counts = patient['patient_counts']
        require(counts['dfu'] == counts['healed'] + counts['not_healed'] == 11, 'P1-C07: mapped patient unit changed')
        check_claim('P1-C07', patient_source, f"patient-unit difference {effect['healer_minus_nonhealer_mean_difference']:.4f}; exhaustive p={effect['exact_two_sided_permutation_p']:.6f}; add-one p={effect['add_one_two_sided_permutation_p']:.6f}; bootstrap CI [{effect['bootstrap_95_ci'][0]:.4f},{effect['bootstrap_95_ci'][1]:.4f}]; sensitivity margin {patient['equivalence']['exact_boundary']['equivalence_margin_infimum']:.6f}")
        specimen_source = base + 'patient_mapping_equivalence/report.json'
        specimen = report(specimen_source)
        effect_sample = specimen['effect']
        check_claim('P1-C08', specimen_source, f"sample-unit difference {effect_sample['healer_minus_nonhealer_mean_difference']:.4f}; exhaustive p={effect_sample['exact_two_sided_permutation_p']:.6f}; add-one p={effect_sample['add_one_two_sided_permutation_p']:.6f}; margin {specimen['equivalence']['exact_boundary']['equivalence_margin_infimum']:.6f}", ('not a clinical MCID',))
        math_source = base + 'mathematical_audit/report.json'
        audit = report(math_source)
        state, permutation = (audit['state_decomposition'], audit['patient_permutation'])
        extreme, allocations = (permutation['extreme_allocations'], permutation['allocations'])
        check_claim('P1-MATH', math_source, f"Residual RMS {state['specimen_weighted_residual_rms']:.7f}; {state['empty_bin_specimens']} empty high bins; exact p={extreme}/{allocations}={extreme / allocations:.6f}; add-one p={extreme + 1}/{allocations + 1}={(extreme + 1) / (allocations + 1):.6f}")
        for unit, contrast in (('patient', effect), ('specimen', effect_sample)):
            perm = audit[f'{unit}_permutation']
            total, extreme = (perm['allocations'], perm['extreme_allocations'])
            for field, expected in (('exact_probability', extreme / total), ('recorded_add_one_probability', (extreme + 1) / (total + 1))):
                require(math.isclose(perm[field], expected, abs_tol=1e-12, rel_tol=0), f'P1-MATH: {unit} permutation arithmetic differs')
            require(perm == contrast['permutation'], f'P1-MATH: {unit} audit differs from contrast report')
        methods = ('topic_simplex_theta0', 'module_score', 'pca', 'nmf')
        inference_source = base + 'representation_inference/report.json'
        inference = report(inference_source)
        auc = [inference['auc'][method]['loso_auc'] for method in methods]
        require(len(inference['pairwise_auc_difference']) == 6 and all((row['ci_excludes_zero'] is False and row['bootstrap_95_ci'][0] <= 0 <= row['bootstrap_95_ci'][1] for row in inference['pairwise_auc_difference'].values())), 'P1-C16: pairwise ranking uncertainty changed')
        require(inference['projection_revision']['sample_bootstrap_invariance_proof']['old_new_cross_arm_pair_credits_identical'] is True, 'P1-C16: inherited pairwise intervals lack rank-invariance support')
        check_claim('P1-C16', inference_source, f'AUC topic {auc[0]:.3f}; module {auc[1]:.3f}; PCA {auc[2]:.3f}; NMF {auc[3]:.3f}; no pairwise interval excludes zero', ('were reused',))
        benchmark_source = base + 'patient_unit_representation_benchmark/report.json'
        benchmark = report(benchmark_source)
        auc = [benchmark['auc'][method]['auc'] for method in methods]
        p_values = '/'.join((f"{benchmark['permutation_null'][method]['p_one_sided']:.4f}" for method in methods))
        check_claim('P1-C23', benchmark_source, f'11-patient leave-one-out AUCs: Topic {auc[0]:.3f}, module {auc[1]:.3f}, PCA {auc[2]:.3f}, NMF {auc[3]:.3f}; permutation p={p_values}', ('not clinical ranking or independent validation', 'OOF scores and permutation nulls were reused', 'historical diagnostic', 'not calibrated cross-patient discrimination', 'add-one'))
        require(benchmark['projection_revision']['reused_non_topic_oof_scores']['methods'] == list(methods[1:]), 'P1-C23: historical non-topic OOF provenance changed')
        semantic = benchmark['patient_semantics']['patient_collapsed_unit']
        check_claim('P1-C25', benchmark_source, f"patient-collapsed semantic rho={semantic['spearman_rho']:.3f} across {semantic['n_patients']} mapped patients")
        correlations_source = base + 'representation_benchmark/report.json'
        correlations = report(correlations_source)['cross_method_spearman_representative']
        require(len(correlations) == 6, 'P1-C27: incomplete cross-method correlation pairs')
        check_claim('P1-C27', correlations_source, f'cross-method sample-score Spearman correlations {min(correlations.values()):.3f}–{max(correlations.values()):.3f}')
        robustness_source = base + 'robustness_extensions/report.json'
        robustness = report(robustness_source)['patient']
        semantics, influence = (robustness['patient_semantics'], robustness['patient_outcome_influence'])
        check_claim('patient_robustness_extension', robustness_source, f"patient semantic rho {semantics['rho']:.4f}; bootstrap CI [{semantics['bootstrap_95_interval'][0]:.4f},{semantics['bootstrap_95_interval'][1]:.4f}]; omission rho {semantics['omission_range'][0]:.4f}–{semantics['omission_range'][1]:.4f}; all-cell difference {influence['observed_difference']:.4f}; omission difference {influence['omission_range'][0]:.4f}–{influence['omission_range'][1]:.4f}")
        projection_source = base + 'report.json'
        projection = report(projection_source)
        identity = projection['source_identity']
        require(identity['raw_identity_included'] is True and all((identity['order_contract'][key] is True for key in ('obs_gsm_equal', 'obs_arm_equal', 'unique_cell_ids'))), 'P1-REV-MEAS: raw identity/order contract is not verified')
        require(all((projection['verification'].get(key) is True for key in ('all_source_hashes_unchanged', 'finite_json', 'expected_counts', 'topic_sample_bootstrap_pair_invariance', 'state_identity_error_below_1e12'))), 'P1-REV-MEAS: projection verification failed')
        check_claim('P1-REV-MEAS', projection_source, f"deterministic softmax(mu) projection; {identity['n_cells']} cells; raw identity/order contract verified")
        repeat_source = base + 'repeat_specimens/report.json'
        repeat = report(repeat_source)
        pair = next((row for row in repeat['paired_specimens'] if (row['sample_a'], row['sample_b']) == ('G7', 'G8')))
        require(pair['absolute_mean_difference'] > max(pair['absolute_mean_difference_split_half_a_upper975'], pair['absolute_mean_difference_split_half_b_upper975']), 'P1-REV-REPEAT: repeat/split-half comparison changed')
        check_claim('P1-REV-REPEAT', repeat_source, f"{repeat['n_repeated_pairs']} repeat pairs, {repeat['n_dfu_repeated_patients']} DFU; G7/G8 absolute mean difference {pair['absolute_mean_difference']:.4f} and CDF supremum {pair['ecdf_supremum']:.4f}; split-half upper limits {pair['absolute_mean_difference_split_half_a_upper975']:.4f}/{pair['absolute_mean_difference_split_half_b_upper975']:.4f}; same-day repeat differences can exceed split-half sampling ranges", ('not biological confidence intervals',))
        label_source = revision + 'measurement_controls/dfu_label_report.json'
        label = report(label_source)['specimen_test']
        fraction = next((row for row in label['readouts'] if row['readout'] == 'high_state_fraction'))
        check_claim('P1-C13', label_source, f"DFU-only specimen high-state-fraction difference {fraction['difference']:+.4f}; {label['allocations']} allocations; exact p={fraction['exact_p']:.4f}; 95% null interval [{fraction['null_percentile_025']:+.4f},{fraction['null_percentile_975']:+.4f}]", ('not the primary patient test',))
        shuffle_source = revision + 'measurement_controls/raw_shuffle_report.json'
        shuffle = report(shuffle_source)
        coords = shuffle['coordinate_summary']
        check_claim('P1-C14', shuffle_source, f"Raw-count gene-wise shuffle: mean perplexity {coords['real_counts']['perplexity_mean']:.2f}/15 to {coords['column_shuffled_counts']['perplexity_mean']:.2f}/15; historical concentration cutoff {shuffle['interpretation']['historical_concentration_cutoff']:.2f} not exceeded; topology audit not run", ('did not establish failure', 'library sizes were not preserved'))
        require(shuffle['interpretation']['shuffled_mean_perplexity_exceeds_historical_cutoff'] is False and shuffle['interpretation']['topology_audit_run'] is False, 'P1-C14: historical integrity-gate interpretation has changed')
        design_source = base + 'design_power_simulation/report.json'
        design = report(design_source)
        difference = design['design_a_detect_difference']['difference_0.03']['total']
        equivalence = design['design_b_declare_equivalence']['margin_0.05']['total']
        check_claim('P1-C22', design_source, f'candidate difference 0.03 requires {difference} total hypothetical independent observations; candidate equivalence margin 0.05 requires {equivalence} at approximately 80% target power', ('not findings', 'not define the clinical MCID'))

    def check_population() -> None:
        historical_source = revision + 'population/historical_benchmark/report.json'
        historical = report(historical_source)
        donor_rows = historical['donor_summary']
        better = '/'.join(sorted((row['held_donor'] for row in donor_rows if row['matched_equal_donor_mean_ed'] < row['shared_cfm_ed'])))
        worse = '/'.join(sorted((row['held_donor'] for row in donor_rows if row['matched_equal_donor_mean_ed'] >= row['shared_cfm_ed'])))
        shared, marginal = (historical['original_shared_cfm_mean_ed'], historical['matched_equal_donor']['mean'])
        require(len(donor_rows) == 3 and len({row['held_donor'] for row in donor_rows}) == 3, 'P2-C05: historical donor coverage changed')
        check_claim('P2-C05', historical_source, f'historical shared CFM mean ED {shared:.4f}; target-marginal matched mean ED {marginal:.4f}; target-marginal lower for {better} but not {worse}', ('not universally',))
        check_claim('P2-REV-MARGINAL', historical_source, f'matched target-marginal mean ED {marginal:.4f} versus historical shared {shared:.4f}; {worse} favors shared field', ('challenges a universal',))
        conditioning_source = 'outputs/analysis/donor_conditioned_benchmark/report.json'
        conditioning = report(conditioning_source)
        rows = list(conditioning['in_sample'].values())
        gains = sum((row['conditioned'] < row['shared'] for row in rows))
        means = conditioning['summary']
        require(means['M4_conditioned_cfm']['mean_ed'] > means['M1_shared_cfm']['mean_ed'], 'P2-C05-CONDITIONING: historical held-out comparison changed')
        check_claim('P2-C05-CONDITIONING', conditioning_source, f"historical conditioned in-sample gains {gains}/{len(rows)}; held-out mean ED conditioned {means['M4_conditioned_cfm']['mean_ed']:.4f} versus shared {means['M1_shared_cfm']['mean_ed']:.4f}", ('historical comparison', 'does not identify the correct model class'))
        curve_source = revision + 'population/report.json'
        population = report(curve_source)
        curve = population['curve']
        shared = next((row for row in curve['summary'] if row['method'] == 'Shared CFM'))
        displacement = next((row for row in curve['summary'] if row['method'] == 'Mean displacement'))
        stochasticity = curve['training_stochasticity']
        require(population['status'] == 'completed' and population['independent_new_data'] is False and (population['biological_donors'] == 3) and (population['seeds'] == [0, 1, 2]), 'P2-C06: common-reference scope changed; seeds are not biological donors')
        verification = population['verification']
        require(verification['status'] == 'passed' and verification['saved_fit_count'] == population['fits'] and all((verification[key] is True for key in ('unique_cell_ids', 'training_donor_exclusion', 'reference_excludes_held_donor', 'source_target_ids_checked'))), 'P2-C06: common-reference fit/exclusion checks failed')
        for row in (shared, displacement):
            require(math.isclose(row['k1_mean_ed'] - row['k2_mean_ed'], row['mean_reduction'], abs_tol=1e-12, rel_tol=0), 'P2-C06: common-reference reduction arithmetic differs')
        check_claim('P2-C06', curve_source, f"common-reference {population['fits']}-fit shared CFM k=1 mean ED {shared['k1_mean_ed']:.4f}; k=2 mean ED {shared['k2_mean_ed']:.4f}; mean reduction {shared['mean_reduction']:.4f}")
        check_claim('P2-C07', curve_source, f"common-reference seed reductions {shared['seed_reduction_min']:.4f}-{shared['seed_reduction_max']:.4f}; max configuration seed range {stochasticity['maximum_configuration_range']:.4f}; mean within-configuration SD {stochasticity['mean_configuration_sd']:.4f}")
        check_claim('P2-C10', curve_source, f"common-reference donor-block reduction {shared['mean_reduction']:.4f}; descriptive range [{shared['descriptive_block_range_lower']:.4f},{shared['descriptive_block_range_upper']:.4f}]; plateau unsupported", ('cannot identify a plateau or required cohort size',))
        check_claim('P2-C12', curve_source, f"common-reference mean-displacement k=1 ED {displacement['k1_mean_ed']:.4f}, k=2 ED {displacement['k2_mean_ed']:.4f}; reduction {displacement['mean_reduction']:.4f}", ('not unique to the neural field',))
        check_claim('P2-REV-COMMONREF', curve_source, f"{population['fits']}-fit common-reference shared CFM {shared['k1_mean_ed']:.4f} to {shared['k2_mean_ed']:.4f}; {shared['donor_blocks_improved']}/{shared['n_donor_blocks']} donor blocks improved")
        ablation_source = revision + 'source_ablation/report.json'
        ablation = report(ablation_source)
        checks, macro = (ablation['checks'], ablation['macro_equal_donor'])
        require(checks['n_independent_biological_donors'] == 3 and ablation['fits'] == 9 and (ablation['draws_per_fit'] == 50) and all((checks[key] is True for key in ('no_new_fit', 'input_manifest_verified', 'fit_checkpoint_hashes_verified', 'source_target_identity_verified', 'real_score_reproduced'))), 'P2-REV-SOURCE-ABLATION: saved-fit or biological scope changed')
        real_better = '/'.join(sorted((row['held_donor'] for row in ablation['per_donor'] if row['real_ed'] < row['wrong_ed'])))
        require(len(ablation['per_donor']) == 3, 'P2-REV-SOURCE-ABLATION: incomplete donor coverage')
        check_claim('P2-REV-SOURCE-ABLATION', ablation_source, f"k=2, three-seed mean ED real held source {macro['real_ed']:.4f}, substituted training Wound1 {macro['wrong_ed']:.4f}, training Wound7 marginal {macro['marginal_ed']:.4f}; only {real_better} favors real over substituted mean", ("consistent incremental value of the new donor's source has not been established",))

    def check_pair_discrimination() -> None:
        source = 'outputs/scientific_revision_20260923/patient_pair_discrimination/report.json'
        pair_report = report(source)
        protocol = pair_report['protocol']
        pairs = protocol['held_pairs']
        require(pair_report['status'] == 'complete' and protocol['patients'] == 11 and (pairs == protocol['healed'] * protocol['nonhealed'] == 28) and (protocol['training_patients_per_fold'] == protocol['patients'] - 2) and (protocol['training_healed'] == protocol['healed'] - 1) and (protocol['training_nonhealed'] == protocol['nonhealed'] - 1), 'P1-PAIR-DISCRIMINATION: patient/pair/training unit changed')
        require(protocol['nmf_max_iter'] == 10000 and pair_report['nmf_warning_folds'] == [] and ('no held-out criterion used' in protocol['nmf_iteration_note']), 'P1-PAIR-DISCRIMINATION: final convergence protocol differs from the completed run')
        folder = Path(source).parent
        for name in ('pair_scores.csv', 'folds.json'):
            require(hashlib.sha256((ROOT / folder / name).read_bytes()).hexdigest() == pair_report['output_sha256'][name], f'P1-PAIR-DISCRIMINATION: changed {name}')
        folds = report(str(folder / 'folds.json'))
        require(len(folds) == pairs and {row['fold'] for row in folds} == set(range(pairs)), 'P1-PAIR-DISCRIMINATION: incomplete fold set')
        for row in folds:
            train, held = (set(row['training_ids']), set(row['held_ids']))
            require(len(train) == 9 and len(held) == 2 and (not train & held) and (row['held_outcomes'] == [1, 0]) and (row['states']['nmf']['warnings'] == []) and all((row[key] is True for key in ('positive_affine_credit_invariant', 'positive_nmf_factor_scale_invariant', 'scoring_does_not_mutate_fit'))), 'P1-PAIR-DISCRIMINATION: same-model isolation/convergence contract failed')
        scores = list(csv.DictReader((ROOT / folder / 'pair_scores.csv').read_text().splitlines()))
        methods = ('topic_simplex_theta0', 'module_score', 'pca', 'nmf')
        summaries = {row['method']: row for row in pair_report['summary']}
        require(set(summaries) == set(methods) and len(pair_report['summary']) == len(methods) and (len(scores) == pairs * len(methods)), 'P1-PAIR-DISCRIMINATION: incomplete method coverage')
        for method in methods:
            rows = [row for row in scores if row['method'] == method]
            summary = summaries[method]
            require(len(rows) == pairs and len({(r['held_healed_id'], r['held_nonhealed_id']) for r in rows}) == pairs and ({int(row['fold']) for row in rows} == set(range(pairs))), f'P1-PAIR-DISCRIMINATION: incomplete pairs for {method}')
            credits = []
            for row in rows:
                a, b = (float(row['healed_score']), float(row['nonhealed_score']))
                require(math.isfinite(a) and math.isfinite(b), 'P1-PAIR-DISCRIMINATION: nonfinite pair score')
                credit = 1.0 if a > b else 0.5 if a == b else 0.0
                require(float(row['credit']) == credit, 'P1-PAIR-DISCRIMINATION: incorrect pair credit')
                credits.append(credit)
            require(summary['n_pairs'] == pairs and summary['wins'] == credits.count(1.0) and (summary['ties'] == credits.count(0.5)) and (summary['losses'] == credits.count(0.0)) and (summary['credit_sum'] == sum(credits)) and math.isclose(summary['auc'], sum(credits) / pairs, rel_tol=0, abs_tol=1e-12), f'P1-PAIR-DISCRIMINATION: {method} summary differs from saved pair credits')
            require(not any((re.search('(^p$|p_value|confidence|bootstrap|(^|_)ci($|_))', key, re.I) for key in summary)), 'P1-PAIR-DISCRIMINATION: unsupported p value or confidence interval')
        credits = '/'.join((f"{summaries[method]['credit_sum']:g}" for method in methods))
        aucs = '/'.join((f"{summaries[method]['auc']:.6f}" for method in methods))
        nmf = summaries['nmf']
        check_claim('P1-PAIR-DISCRIMINATION', source, f"{pairs} same-model held pairs from {protocol['patients']} patients; {protocol['training_patients_per_fold']} training patients per fold; credit sums Topic/module/PCA/NMF {credits}; discrimination {aucs}; NMF {nmf['wins']} wins, {nmf['ties']} ties, {nmf['losses']} losses; iteration cap {protocol['nmf_max_iter']}; final warning folds {len(pair_report['nmf_warning_folds'])}", ('not 28 independent biological replicates', 'no p values or confidence intervals', 'does not establish independent validation', 'not selected using held-pair discrimination'))

    def check_mixture() -> None:
        source = 'outputs/scientific_revision_20260923/population_mixture/report.json'
        mixture = report(source)
        verification, protocol = (mixture['verification'], mixture['protocol'])
        require(mixture['status'] == 'completed' and mixture['biological_donors'] == 3 and (mixture['new_biological_donors'] == 0) and (mixture['new_model_training'] is False) and (mixture['independent_new_data'] is False) and (verification['status'] == 'passed') and all((verification[key] is True for key in ('relevant_input_manifests_verified', 'fit_grid_verified', 'donor_and_time_exclusions_verified', 'scalers_recalculated', 'shared_target_and_cross_seed_draw_indices', 'no_new_training'))), 'P2-MIXTURE-IDENTITY: saved-input/no-new-donor scope changed')
        exact = mixture['exact_marginal']
        require(len(exact['per_fold']) == 3 and exact['macro']['held_donors'] == 3, 'P2-MIXTURE-IDENTITY: incomplete marginal donor coverage')
        for row in [*exact['per_fold'], exact['macro']]:
            gain = row['individual_mean_ed'] - row['mixed_ed']
            rhs = row['component_pair_ed'] / 4
            require(math.isclose(gain, rhs, rel_tol=0, abs_tol=1e-12) and math.isclose(row['convexity_gain'], gain, rel_tol=0, abs_tol=1e-12) and math.isclose(row['quarter_component_pair_ed'], rhs, rel_tol=0, abs_tol=1e-12), 'P2-MIXTURE-IDENTITY: component/mixture arithmetic differs')
        require(max((abs(row['identity_residual']) for row in exact['per_fold'])) == exact['macro']['maximum_absolute_identity_residual'] < 1e-12 and verification['maximum_sampled_identity_residual'] < 1e-12 and (verification['maximum_donor_mass_error'] < 1e-12), 'P2-MIXTURE-IDENTITY: identity residual or donor mass check failed')
        macro = exact['macro']
        check_claim('P2-MIXTURE-IDENTITY', source, f"exact equal-donor target marginal mean ED {macro['individual_mean_ed']:.6f} to {macro['mixed_ed']:.6f}; reduction {macro['convexity_gain']:.6f} = quarter component-pair ED {macro['quarter_component_pair_ed']:.6f}; maximum identity residual {macro['maximum_absolute_identity_residual']:.2e}; {macro['held_donors']} held donors", ('arithmetic identity is verified', 'does not establish target-specific information', 'does not decompose gains from jointly fitted flow fields', 'No new biological donors, p values or confidence intervals'))
        comparison = mixture['source_comparison']
        macro = comparison['macro']
        rows = {row['held_donor']: row for row in comparison['per_donor']}
        require(len(rows) == 3 and len(comparison['per_donor']) == 3 and (len(comparison['per_fit']) == macro['saved_fits'] == 9) and (protocol['requested_prediction_cells'] == 200) and (protocol['draws_per_fit'] == macro['draws_per_fit'] == 50), 'P2-INDIVIDUAL-SOURCE: matched-particle/saved-fit design changed')
        for row in [*rows.values(), macro]:
            require(math.isclose((row['wrong_a_ed'] + row['wrong_b_ed']) / 2, row['individual_wrong_mean_ed'], rel_tol=0, abs_tol=1e-12) and math.isclose(row['mixture_component_mean_ed'] - row['mixed_wrong_ed'], row['mixture_component_pair_ed'] / 4, rel_tol=0, abs_tol=1e-12) and math.isclose(row['mixture_identity_gain'], row['mixture_component_pair_ed'] / 4, rel_tol=0, abs_tol=1e-12), 'P2-INDIVIDUAL-SOURCE: individual/mixture-subset arithmetic differs')
        for key in ('real_ed', 'individual_wrong_mean_ed', 'mixed_wrong_ed'):
            require(math.isclose(macro[key], sum((row[key] for row in rows.values())) / len(rows), rel_tol=0, abs_tol=1e-12), 'P2-INDIVIDUAL-SOURCE: macro must give equal weight to donors')
        real_better = '/'.join(sorted((d for d, row in rows.items() if row['real_ed'] < min(row['wrong_a_ed'], row['wrong_b_ed']))))
        mixed_better = '/'.join(sorted((d for d, row in rows.items() if row['mixed_wrong_ed'] < row['real_ed'])))
        p28 = rows['PWH28']
        check_claim('P2-INDIVIDUAL-SOURCE', source, f"{protocol['requested_prediction_cells']}-particle audit of {macro['saved_fits']} saved fits, {macro['draws_per_fit']} draws per fit; mean ED real {macro['real_ed']:.4f}, mean individual substituted {macro['individual_wrong_mean_ed']:.4f}, mixed substituted {macro['mixed_wrong_ed']:.4f}; PWH28 real {p28['real_ed']:.4f}, individual A/B {p28['wrong_a_ed']:.4f}/{p28['wrong_b_ed']:.4f}, mixed {p28['mixed_wrong_ed']:.4f}; real lower than both individual alternatives for {real_better}; mixed lower than real for {mixed_better}", ('is not the exact convexity identity', 'do not establish universal absence of source-state value', 'no p values or confidence intervals'))
    try:
        if 'measurement_construct' in papers:
            check_measurement()
            check_pair_discrimination()
            check_full_depth()
        if 'population_dynamics' in papers:
            check_population()
            check_mixture()
            check_paired_clock()
            check_observable()
    except (OSError, ValueError, KeyError, TypeError, StopIteration, ZeroDivisionError) as error:
        fail(f'revised report missing or incompatible with claim contract: {error}')
    print(f'PASS  current manuscript titles and {len(checked)} revised/historical-separated claims match source reports')

def check_revision_tables(manuscript_text: dict[str, str]) -> None:
    """Keep the new estimands' table labels and uncertainty distinct from history."""
    for key, table, equation in (('measurement_construct', 'tab:p1pairauc', 'eq:p1pairauc'), ('population_dynamics', 'tab:p2mixtureidentity', 'eq:p2mixture'), ('population_dynamics', 'tab:p2mixtureaudit', 'eq:p2mixture')):
        if key not in manuscript_text:
            continue
        text = manuscript_text[key]
        if any((token not in text for token in (f'\\label{{{table}}}', f'\\ref{{{table}}}', f'\\label{{{equation}}}', f'\\eqref{{{equation}}}'))):
            fail(f'{key}: missing revised table/equation contract: {table}, {equation}')
        tables = [block for _, block in re.findall('\\\\begin\\{(longtable|table\\*?)\\}(.*?)\\\\end\\{\\1\\}', text, re.S) if f'\\label{{{table}}}' in block]
        if len(tables) != 1:
            fail(f'{table}: expected one revised-results table')
        for block in tables:
            if re.search('(?i)(?:\\bp\\s*[=<]|\\bCI\\s*[=:]|confidence\\s+interval\\s*[=:])\\s*\\$?\\s*\\d', block):
                fail(f'{table}: unsupported p value or confidence interval for revised descriptive estimand')
            if re.search('(?i)&\\s*(?:\\$?p(?:[- ]value)?\\$?|(?:95\\\\?%\\s*)?CI|confidence interval)\\s*(?:&|\\\\\\\\)', block):
                fail(f'{table}: unsupported inference column for revised descriptive estimand')
            if re.search('[\\[(]\\s*[+-]?\\d*\\.\\d+\\s*,\\s*[+-]?\\d*\\.\\d+\\s*[\\])]', block):
                fail(f'{table}: unsupported interval for revised descriptive estimand')
        block = tables[0]
        if key == 'measurement_construct':
            report = json.loads((ROOT / 'outputs/scientific_revision_20260923/patient_pair_discrimination/report.json').read_text())
            expected_rows = [[row['label'], str(row['wins']), str(row['ties']), str(row['losses']), f"{row['credit_sum']:.1f} / {row['n_pairs']}", f"{row['auc']:.4f}"] for row in report['summary']]
        elif table == 'tab:p2mixtureidentity':
            report = json.loads((ROOT / 'outputs/scientific_revision_20260923/population_mixture/report.json').read_text())
            fields = ('individual_mean_ed', 'mixed_ed', 'convexity_gain', 'quarter_component_pair_ed')
            expected_rows = [[row['held_donor'], *(f'{row[field]:.6f}' for field in fields)] for row in report['exact_marginal']['per_fold']]
            macro = report['exact_marginal']['macro']
            expected_rows.append(['Three-donor mean', *(f'{macro[field]:.6f}' for field in fields)])
        else:
            report = json.loads((ROOT / 'outputs/scientific_revision_20260923/population_mixture/report.json').read_text())
            expected_rows = [[row['held_donor'], f"{row['real_ed']:.4f}", f"{row['wrong_a_donor']}: {row['wrong_a_ed']:.4f}", f"{row['wrong_b_donor']}: {row['wrong_b_ed']:.4f}", f"{row['mixed_wrong_ed']:.4f}"] for row in report['source_comparison']['per_donor']]
            macro = report['source_comparison']['macro']
            expected_rows.append(['Three-donor mean', *(f'{macro[field]:.4f}' for field in ('real_ed', 'wrong_a_ed', 'wrong_b_ed', 'mixed_wrong_ed'))])
        actual_rows = [[cell.strip() for cell in line.split('\\\\', 1)[0].split('&')] for line in block.splitlines() if '&' in line and '\\caption' not in line]
        for expected in expected_rows:
            matching = [row for row in actual_rows if row[0] == expected[0]]
            if matching != [expected]:
                fail(f'{table}: {expected[0]} row differs from revised report')
    print('PASS  revised table/equation labels, report-derived rows and descriptive-only reporting')

def check_latest_tables(manuscript_text: dict[str, str]) -> None:
    """Check every numeric cell in the completed depth/clock/expression tables."""
    audit = ROOT / 'outputs/scientific_revision_20260925'
    expected = {}
    fmt = lambda value, digits=4: f'{float(value):.{digits}f}'
    if 'measurement_construct' in manuscript_text:
        depth = json.loads((audit / 'full_pipeline_depth/full/report.json').read_text())
        expected['tab:p1fulldepth'] = [[fmt(100 * r['rate'], 0) + '%', 'Baseline' if r['seed'] == -1 else str(r['seed']), str(r['current_qc']), f"{r['lost_qc']} / {r['gained_qc']}", fmt(100 * r['common_qc_lineage_change_fraction'], 2) + '%', str(r['current_qc_zero_marker_cells'])] for r in sorted(depth['pooled_qc_gate_by_run'], key=lambda r: (-r['rate'], r['seed']))]
        names = {'all_cell_topic0_mean': 'All-cell mean', 'fibroblast_topic0_mean': 'Fibroblast mean', 'fibroblast_high_state_fraction': 'High-state fraction', 'fibroblast_fraction': 'Fibroblast composition'}
        expected['tab:p1depthoutcomes'] = [[fmt(100 * r['rate'], 0) + '%', names[r['readout']], f"{r['n_estimable']} / {r['n_runs']}", fmt(r['difference_min']) + ' to ' + fmt(r['difference_max']), fmt(r['exact_p_min']) + ' to ' + fmt(r['exact_p_max'])] for r in sorted(depth['full_pipeline_outcome_seed_ranges'], key=lambda r: (-r['rate'], list(names).index(r['readout'])))]
    if 'population_dynamics' in manuscript_text:
        clock = json.loads((audit / 'paired_clock/run/paired_clock/report.json').read_text())
        records = list(csv.DictReader((audit / 'paired_clock/run/paired_clock/per_donor_seed.csv').read_text().splitlines()))
        rows = []
        for r in clock['summary']:
            selected = [v for v in records if v['axis'] == r['axis']]
            seed_means = [sum((float(v['flow_ed']) for v in selected if int(v['seed']) == seed)) / 3 for seed in (0, 1, 2)]
            change = max((abs(float(v['flow_ed']) - float(v['refined_flow_ed'])) for v in selected))
            rows.append([r['axis'].replace('_', ' '), fmt(r['flow_ed']), fmt(min(seed_means)) + ' to ' + fmt(max(seed_means)), fmt(r['centroid_ed']), fmt(r['unchanged_source_ed']), f'{change:.2e}'])
        expected['tab:p2pairedclock'] = rows
        observable = json.loads((audit / 'population_observable/final-01/report.json').read_text())
        names = {'shared_cfm_decoded': 'Decoded shared flow', 'persistence_decoded': 'Decoded persistence', 'mean_displacement_decoded': 'Decoded mean displacement', 'training_target_marginal_decoded_exact': 'Decoded training marginal', 'persistence_observed': 'Observed persistence', 'training_target_marginal_observed_exact': 'Observed training marginal'}
        rows = []
        for target in ('original_evaluation', 'all_day7'):
            for k in (1, 2):
                for method, label in names.items():
                    r = next((v for v in observable['macro'] if (v['target_set'], v['k'], v['method']) == (target, k, method)))
                    rows.append(['Saved target' if target == 'original_evaluation' else 'All day 7', str(k), label, fmt(r['total_variation']), fmt(r['js_divergence_nats']), fmt(r['gene_rmse'], 6)])
        expected['tab:p2observable'] = rows
        expected['tab:p2decoderreference'] = [[r['held_donor'], 'Saved target' if r['target_set'] == 'original_evaluation' else 'All day 7', str(r['n_target']), fmt(r['total_variation']), fmt(r['js_divergence_nats']), fmt(100 * r['missing_panel_mass_mean'], 2) + '%'] for r in observable['decoder_reference']]
    for label, rows in expected.items():
        text = manuscript_text['measurement_construct' if label.startswith('tab:p1') else 'population_dynamics']
        blocks = [block for block in re.findall('\\\\begin\\{longtable\\}(.*?)\\\\end\\{longtable\\}', text, re.S) if f'\\label{{{label}}}' in block]
        if len(blocks) != 1 or f'\\ref{{{label}}}' not in text:
            fail(f'{label}: missing unique table or Results citation')
        body = blocks[0].split('\\endfoot', 1)[-1].replace('\\%', '%')
        actual = [[cell.strip() for cell in line.split('\\\\', 1)[0].split('&')] for line in body.splitlines() if '&' in line]
        if actual != rows:
            fail(f'{label}: numeric table rows differ from completed report')
    print(f'PASS  {len(expected)} completed-audit tables: every numeric cell and row matches its source')

def check_result_coverage(ledger: dict, coverage: dict) -> None:
    """Verify report hashes and coverage without reclassifying historical results."""
    entries = {}
    for entry in coverage['entries']:
        source = entry['source']
        if source in entries:
            fail(f'duplicate coverage source: {source}')
        entries[source] = entry
        path = ROOT / source
        if not path.is_file():
            fail(f'coverage source missing: {source}')
        digest = entry.get('sha256', '')
        if not re.fullmatch('[0-9a-f]{64}', digest) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            fail(f'coverage SHA256 mismatch: {source}')
    for source, entry in entries.items():
        replacement = entry.get('superseded_by')
        if replacement and (replacement not in entries or replacement == source or (not entry.get('current_use'))):
            fail(f'coverage replacement/current-use boundary missing: {source}')
    referenced = set()
    for paper in ledger['papers'].values():
        for claim in paper['claims']:
            source = claim.get('source')
            if source not in entries:
                fail(f"{claim['id']}: ledger source has no hashed coverage entry: {source}")
            entry = entries[source]
            if entry['status'] in {'superseded', 'diagnostic_only'} or entry.get('current_use') == 'historical_only':
                fail(f"{claim['id']}: historical-only report cannot support a current ledger claim: {source}")
            referenced.add(source)
    print(f'PASS  report coverage: {len(entries)} verified hashes; {len(referenced)} ledger sources covered; historical descriptors preserved')


def check_export_reports():
    import hashlib
    index=json.loads((ROOT/'RESULTS_INDEX.json').read_text())
    for entry in index:
        path=ROOT/entry['report']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            fail('Missing or changed scientific report: '+entry['report'])
        json.loads(path.read_text())
    provenance=json.loads((ROOT/'EXPORT_PROVENANCE.json').read_text())
    exported={entry['exported']:entry for entry in provenance['files']}
    for entry in provenance['files']:
        if entry.get('original'):
            path=ROOT/entry['original']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['source_sha256']:
                fail('Original source report changed: '+entry['original'])
    # Pair-audit reports bind recursive fitted states and their pilot as
    # producing bytes. Some textual artifacts have translated paths; check
    # their exact native originals as well as the current exported bytes.
    for path in (ROOT/'outputs/scientific_revision_20260923').glob('patient_pair_discrimination/**/report.json'):
        for name,digest in json.loads(path.read_text()).get('output_sha256',{}).items():
            artifact=path.parent/name
            relative=str(artifact.relative_to(ROOT))
            record=exported.get(relative)
            actual=hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact.is_file() else None
            if actual==digest:
                continue
            if (record is None or 'original' not in record or record['source_sha256']!=digest
                    or actual!=record['exported_sha256']
                    or hashlib.sha256((ROOT/record['original']).read_bytes()).hexdigest()!=digest):
                fail('Missing or changed pair-audit artifact: '+relative)
    for path in (ROOT/'outputs').glob('scientific_revision_*/**/output_manifest.json'):
        for name,digest in json.loads(path.read_text()).items():
            artifact=path.parent/name
            if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest()!=digest:
                fail('Missing or changed recursive artifact: '+str(artifact.relative_to(ROOT)))
    for pattern in ('scientific_revision_*/**/preparation_manifest.json','scientific_revision_*/**/artifact_manifest.json'):
        for path in (ROOT/'outputs').glob(pattern):
            entries=json.loads(path.read_text())
            if path.name=='artifact_manifest.json':entries=entries['files_sha256']
            for name,digest in entries.items():
                artifact=path.parent/name
                if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest()!=digest:
                    fail('Missing or changed recursive artifact: '+str(artifact.relative_to(ROOT)))
    revision=ROOT/'outputs/scientific_revision_20260922/population'
    if revision.is_dir():
        report=json.loads((revision/'report.json').read_text())
        text=read_native(next(iter(PAPER_FILES.values())))
        curve=next(row for row in report['curve']['summary'] if row['method']=='Shared CFM')
        for value in (curve['k1_mean_ed'],curve['k2_mean_ed'],curve['mean_reduction']):
            if f'{value:.4f}' not in text:
                fail('Current donor-curve report differs from manuscript')
        ablation=json.loads((revision.parent/'source_ablation/report.json').read_text())
        if not ablation['checks']['no_new_fit'] or ablation['checks']['n_independent_biological_donors']!=3:
            fail('Source-information scope changed')
        macro=ablation['macro_equal_donor']
        for key in ('real_ed','wrong_ed','marginal_ed'):
            if f'{macro[key]:.4f}' not in text:
                fail('Current source-information report differs from manuscript')
    print('PASS  current scientific reports, original provenance and recursive artifacts')

def main() -> int:
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not LEDGER_PATH.exists():
        fail('claim ledger is missing')
    ledger = json.loads(LEDGER_PATH.read_text(encoding='utf-8'))
    check_export_reports()
    check_author_style()
    check_revision_claims(ledger)
    if not COVERAGE_PATH.is_file():
        fail('result coverage is missing')
    check_result_coverage(ledger, json.loads(COVERAGE_PATH.read_text(encoding='utf-8')))
    manuscript_text = {}
    for paper, path in PAPER_FILES.items():
        if not path.exists():
            fail(f'{paper}: manuscript is missing: {path.name}')
        manuscript_text[paper] = read_native(path)
    check_revision_tables(manuscript_text)
    check_latest_tables(manuscript_text)
    report_sources = []
    for paper in ledger.get('papers', {}).values():
        for claim in paper.get('claims', []):
            source = claim.get('source')
            if source:
                report_sources.append(source)
    missing_sources = [source for source in report_sources if not (ROOT / source).exists()]
    if missing_sources:
        fail('ledger source missing: ' + ', '.join(sorted(set(missing_sources))))
    for paper, anchors in ANCHORS.items():
        text = manuscript_text[paper]
        missing = [anchor for anchor in anchors if anchor not in text]
        if missing:
            fail(f"{paper}: missing quantitative anchors: {', '.join(missing)}")
    for paper, text in manuscript_text.items():
        for pattern in AFFIRMATIVE_PATTERNS:
            match = pattern.search(text)
            if match:
                excerpt = ' '.join(match.group(0).split())
                fail(f'{paper}: prohibited affirmative claim: {excerpt}')
        for forbidden in ('outputs/', 'scripts/', 'SHA256', 'preflight', 'prompt', 'report.json'):
            if forbidden.lower() in text.lower():
                fail(f'{paper}: internal implementation token leaked: {forbidden}')
    ready_claims = sum((1 for paper in ledger.get('papers', {}).values() for claim in paper.get('claims', []) if claim.get('status') == 'ready'))
    exploratory_claims = sum((1 for paper in ledger.get('papers', {}).values() for claim in paper.get('claims', []) if claim.get('status') == 'exploratory'))
    print(f'PASS  ledger sources={len(set(report_sources))} ready_claims={ready_claims} exploratory_claims={exploratory_claims}')
    print(f'PASS  manuscripts={len(manuscript_text)} quantitative_anchor_sets={len(ANCHORS)}')
    print('PASS  prohibited affirmative claims=0')
    print('PASS  reader-facing drafts contain no repository paths, hashes, gate logs, or prompts')
    print('ALL MANUSCRIPT CHECKS PASSED')
    return 0
if __name__ == '__main__':
    sys.exit(main())
