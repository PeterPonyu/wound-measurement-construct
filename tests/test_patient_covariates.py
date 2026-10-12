import numpy as np
import pandas as pd
import pytest

from scripts.audit_patient_covariates import collapse_covariates, describe_associations


def cells_fixture():
    frame = pd.DataFrame(dict(patient_id=['p1']*5 + ['p2']*3,
        gsm=['g1']*2 + ['g2']*3 + ['g3']*3,
        sample_title=['G1']*2 + ['G2']*3 + ['G3']*3,
        arm=['DFU-healer']*5+['DFU-nonhealer']*3,
        celltype=['fibroblast', 't_cell', 'fibroblast', 'fibroblast', 'fibroblast',
                  'fibroblast', 't_cell', 't_cell'],
        topic_0=[.8, .2, .1, .1, .1, .2, .1, .1]))
    for i in range(1, 15):
        frame[f'topic_{i}'] = (1-frame.topic_0)/14
    technical = pd.DataFrame(dict(gsm=['g1', 'g2', 'g3'], tech_diss=[1, 2, 3],
        tech_umi=[100, 300, 500], tech_ptprc_pos=[.2, .6, .8]))
    return frame, technical


def test_denominator_specific_patient_pooling():
    cells, technical = cells_fixture()
    patients = collapse_covariates(cells, technical, .5).set_index('patient_id')
    assert patients.loc['p1', 'weight'] == .25
    assert patients.loc['p1', 'frac_fibroblast'] == .8
    assert patients.loc['p1', 'tech_umi'] == 250
    assert patients.loc['p1', 'n_cells_total'] == 5
    assert patients.loc['p1', 'n_fibroblasts'] == 4
    assert patients.loc['p1', 'meta_g_number'] == 1.5


def test_duplicate_or_missing_technical_rows_rejected():
    cells, technical = cells_fixture()
    for invalid in (technical.iloc[:-1], pd.concat([technical, technical.iloc[:1]])):
        with pytest.raises(ValueError, match='coverage'):
            collapse_covariates(cells, invalid, .5)


def test_nonfinite_technical_rejected():
    cells, technical = cells_fixture()
    technical.loc[0, 'tech_umi'] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        collapse_covariates(cells, technical, .5)


def test_feature_scope_cannot_silently_change():
    with pytest.raises(ValueError, match='31-feature'):
        describe_associations(pd.DataFrame({'weight': [.1, .2], 'arbitrary': [2, 1]}))
