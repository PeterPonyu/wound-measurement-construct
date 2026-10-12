"""Saved scalar reporting for this study; no model fitting or historical cross-study loader."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

def finite(value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('nonfinite saved scalar')
    return result

def validate_measurement(report: dict) -> None:
    rows = report['fits']
    grid = {(kind, steps, seed) for kind, budgets in (('inherited', (20000, 40000)), ('training', (20000,))) for steps in budgets for seed in (0, 1)}
    keys = [(r['panel_kind'], r['steps'], r['seed']) for r in rows]
    if len(keys) != 6 or set(keys) != grid:
        raise ValueError('topic grid incomplete or duplicated')
    for r in rows:
        if r['panel_size'] != (7002 if r['panel_kind'] == 'inherited' else 7003):
            raise ValueError('topic panel identity differs')
        for key in ('held_objective', 'held_reconstruction', 'equal_patient_training_null', 'within_panel_model_minus_null', 'decoder_cosine_median', 'decoder_cosine_max', 'held_effective_topic_usage', 'historical_reference0_correlation'):
            finite(r[key])
        if not math.isclose(r['held_reconstruction'] - r['equal_patient_training_null'], r['within_panel_model_minus_null'], abs_tol=1e-12):
            raise ValueError('within-panel null contrast differs')
        if not 0 <= r['decoder_cosine_median'] <= r['decoder_cosine_max'] <= 1:
            raise ValueError('invalid decoder cosine')
    if report['new_fits'] != 0 or report['new_biological_units'] != 0 or report['end_to_end_independent'] or report['fit_selection_authorized'] or report['cross_panel_raw_ranking_authorized']:
        raise ValueError('measurement reporting exceeds diagnostic scope')

def fmt(value, digits=5):
    return f'{float(value):.{digits}f}'

def measurement_table_rows(report: dict) -> list[list[str]]:
    validate_measurement(report)
    return [[('Inherited' if r['panel_kind'] == 'inherited' else 'Train-selected') + f" {r['panel_size']}", f"{r['steps'] // 1000}k / {r['seed']}", fmt(r['held_reconstruction']), fmt(r['within_panel_model_minus_null']), fmt(r['decoder_cosine_median'], 3), fmt(r['held_effective_topic_usage'], 3), 'Pass / fail' if r['training_plateau_screen_passed'] else 'Fail / fail'] for r in report['fits']]
