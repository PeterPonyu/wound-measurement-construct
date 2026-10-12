#!/usr/bin/env python3
"""Check the exact native closure and recompute bounded saved-input results."""
import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
STUDY = 1


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def numeric(value, path=''):
    if isinstance(value, dict):
        return {p: v for k, item in value.items() for p, v in numeric(item, path + '/' + k).items()}
    if isinstance(value, list):
        return {p: v for i, item in enumerate(value) for p, v in numeric(item, path + '/' + str(i)).items()}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {path: value}
    return {}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--saved-checks', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.saved_checks and args.output_dir is None:
        parser.error('--saved-checks requires a fresh --output-dir')
    started = time.time()
    closure = json.loads((ROOT/'REPRODUCTION_CLOSURE.json').read_text())
    for name, entry in closure['files'].items():
        path = ROOT/name
        if not path.is_file() or sha(path) != entry['sha256']:
            raise ValueError('Missing or changed closure artifact: ' + name)
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=name)
    sys.path.insert(0, str(ROOT))
    helper = ROOT/f'outputs/followup_20261010/paper{STUDY}/scripts'
    sys.path.insert(0, str(helper))
    receipt = dict(study=closure['study'], file_hashes=len(closure['files']),
                   provenance_gaps=closure['provenance_gaps'], model_training_performed=False,
                   network_access_performed=False, full_refit_performed=False,
                   strict_saved_field_cpu_replay_certified=False)
    if args.smoke or args.saved_checks:
        names = ['core_engines.data_adapter', 'core_engines.topic_simplex_engine',
                 'expansion.common.io', 'expansion.common.rng']
        names += (['expansion.measurement.stability', 'expansion.measurement.worker', 'p1_common']
                  if STUDY == 1 else ['expansion.population.worker', 'expansion.population.methods', 'p2common'])
        for name in names:
            importlib.import_module(name)
        commands = ([['scripts/audit_p1_followup.py', '--help'],
                     ['scripts/55_full_pipeline_depth_audit.py', '--help']]
                    if STUDY == 1 else [['scripts/audit_followup_integration.py', '--help'],
                                        ['scripts/complete_followup_nested.py', '--help'],
                                        ['scripts/external_gse186527.py', '--help']])
        for command in commands:
            subprocess.run([sys.executable, *command], cwd=ROOT, check=True,
                           stdout=subprocess.DEVNULL, env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
        receipt.update(import_smoke='passed', cli_smoke='passed', imported=names, cli_commands=commands)
    if args.saved_checks:
        dest = args.output_dir.resolve()
        if dest == ROOT or not dest.is_relative_to(ROOT):
            raise ValueError('Choose a fresh output directory inside this portable closure')
        dest.mkdir(parents=True, exist_ok=False)
        if STUDY == 1:
            subprocess.run([sys.executable, 'scripts/audit_p1_followup.py', '--output-dir', str(dest/'selection')],
                           cwd=ROOT, check=True)
            selection = json.loads((dest/'selection/report.json').read_text())
            reference = json.loads((ROOT/'output/followup-integration-20261011-r1/p1/audit/report.json').read_text())
            if numeric(selection['topic_search']) != numeric(reference['topic_search']):
                raise ValueError('Selection statistics differ from the saved reference')
            extension = load_module(ROOT/'scripts/run_p1_quantitative_extension.py', 'quantitative_extension')
            extension.OUT = dest/'quantitative'
            extension.MAPPING = ROOT/'output/quantitative-extension-20261011/hugene20_hgnc.tsv'
            if not extension.MAPPING.is_file():
                raise FileNotFoundError('The frozen mapping is required; verification never downloads it')
            extension.main()
            observed = json.loads((extension.OUT/'report.json').read_text())
            saved = json.loads((ROOT/'output/quantitative-extension-20261011/report.json').read_text())
            left, right = numeric(observed), numeric(saved)
            if left.keys() != right.keys():
                raise ValueError('Quantitative result keys differ')
            delta = max(abs(left[k]-right[k]) for k in left)
            if delta > 1e-12:
                raise ValueError('Quantitative results differ: ' + str(delta))
            receipt['saved_result_checks'] = dict(selection_allocations_per_readout=330,
                quantitative_numeric_values=len(left), quantitative_max_abs_error=delta)
        else:
            common = importlib.import_module('p2common')
            common.OUT = dest/'e0'
            common.OUT.mkdir()
            runpy.run_path(str(helper/'e0_reproduce.py'), run_name='__main__')
            current = json.loads((common.OUT/'e0_reproduction.json').read_text())
            if current['harness_reproduction']['rows'] != 2700 or current['harness_reproduction']['missing'] != 0:
                raise ValueError('Deterministic-control reproduction is incomplete')
            if current['harness_reproduction']['max_abs_energy_difference'] > 1e-12:
                raise ValueError('Deterministic-control energies differ')
            # Restore the original results path before checking the declared grid.
            common.OUT = helper.parent/'results'
            grid = importlib.import_module('run_fits')
            declared = [job for stage in ('e1','e2','e3b','e6') for job in grid.jobs(stage)]
            expected = {common.OUT/f"fits/{job['rep']}/{job['task']}/{job['tag']}_s{job['seed']}.npz" for job in declared}
            actual = set((common.OUT/'fits').rglob('*.npz'))
            if actual != expected or len(actual) != 354:
                raise ValueError('Original 354-field grid differs')
            for path in actual:
                if not path.with_name(path.stem+'_field.pt').is_file():
                    raise ValueError('Missing saved field: ' + str(path))
            nested = ROOT/'output/followup-integration-20261011-r1/p2/nested-refits'
            completion = json.loads((nested/'completion.json').read_text())
            if completion['completed'] != 9 or len(list(nested.glob('*/tuned_nested_s*.npz'))) != 9:
                raise ValueError('Selected outer grid is incomplete')
            receipt['saved_result_checks'] = dict(**current['harness_reproduction'],
                original_fields=354, selected_outer_fields=9, table_s25=current['table_s25_full_cell'])
        receipt['output_dir'] = str(dest.relative_to(ROOT))
        receipt['elapsed_seconds'] = time.time()-started
        (dest/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
