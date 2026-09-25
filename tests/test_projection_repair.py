"""Behavioral checks for the deterministic measurement projection contract."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("projection_repair", ROOT / "scripts/repair_measurement_projection.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ProjectionRepairContract(unittest.TestCase):
    def _frames(self):
        projection = pd.DataFrame({
            "gsm": ["G1", "G1", "G2", "G2"],
            "arm": ["DFU-healer", "DFU-healer", "DFU-nonhealer", "DFU-nonhealer"],
            "celltype": ["fibroblast"] * 4,
            **{f"topic_{i}": [1.0 if i == 0 else 0.0] * 4 for i in range(15)},
        })
        obs = pd.DataFrame({"gsm": projection.gsm, "disease": projection.arm})
        mapping = pd.DataFrame({
            "sample_id": ["G1", "G2"], "patient_id": ["P1", "P2"],
            "sample_title": ["G1", "G2"], "healing_status": ["healed", "not_healed"],
            "group": ["DFU-H", "DFU-NH"], "tissue_site": ["foot", "foot"],
        })
        return projection, obs, mapping

    def test_source_order_ids_are_unique_and_are_not_barcodes(self):
        projection, obs, mapping = self._frames()
        cells = MODULE.validated_cells(projection, obs, mapping)
        self.assertEqual(cells.cell_id.tolist(), ["G1:qc_row_0", "G1:qc_row_1", "G2:qc_row_0", "G2:qc_row_1"])
        self.assertNotIn("barcode", cells.cell_id.iloc[0])

    def test_rejects_arm_order_mismatch(self):
        projection, obs, mapping = self._frames()
        obs.loc[0, "disease"] = "DFU-nonhealer"
        with self.assertRaises(ValueError):
            MODULE.validated_cells(projection, obs, mapping)

    def test_rejects_raw_qc_order_mismatch(self):
        projection, obs, mapping = self._frames()
        raw = pd.DataFrame({
            "gsm": projection.gsm, "arm": projection.arm,
            "raw_barcode": ["A", "B", "C", "D"],
            "raw_row_0": [0, 1, 0, 1], "qc_row_0": [1, 0, 0, 1],
        })
        with self.assertRaises(ValueError):
            MODULE.validated_cells(projection, obs, mapping, raw)

    def test_auc_pair_credit_is_invariant_under_monotone_projection(self):
        values = np.array([.1, .2, .3, .4, .5, .6])
        healed = np.array([True, True, True, False, False, False])
        old = MODULE.pair_credit(MODULE.oriented_loo(values, healed), healed)
        new = MODULE.pair_credit(MODULE.oriented_loo(np.exp(values), healed), healed)
        np.testing.assert_array_equal(old, new)

    def test_patient_pooling_retains_distinct_cell_and_specimen_estimands(self):
        cells = pd.DataFrame({
            "gsm": ["A", "A", "B", "C"], "sample_title": ["A", "A", "B", "C"],
            "patient_id": ["P1", "P1", "P1", "P2"],
            "healing_status": ["healed", "healed", "healed", "not_healed"],
            "arm": ["DFU-healer", "DFU-healer", "DFU-healer", "DFU-nonhealer"],
            "topic_0": [.9, .7, .2, .4],
        })
        _, patients = MODULE.aggregate_scores(cells)
        first = patients.set_index("patient_id").loc["P1"]
        self.assertAlmostEqual(first.score_cell_weighted, .6)
        self.assertAlmostEqual(first.score_sample_mean, .5)
        self.assertEqual(first.n_samples, 2)
        self.assertEqual(first.n_cells, 3)

    def test_tied_scores_count_every_exhaustive_auc_allocation(self):
        result, null = MODULE.auc_permutation(np.ones(4), [True, True, False, False])
        self.assertEqual(result["n_relabelings"], 6)
        self.assertEqual(result["extreme_allocations"], 6)
        self.assertEqual(result["exact_probability"], 1.)
        np.testing.assert_array_equal(null, np.full(6, .5))

    def test_exhaustive_effect_keeps_exact_and_add_one_conventions(self):
        result = MODULE.effect(np.array([0.9, 0.8, 0.1, 0.2]), np.array([True, True, False, False]))
        self.assertIn("permutation", result)
        self.assertEqual(result["permutation"]["allocations"], 6)
        self.assertAlmostEqual(result["exact_two_sided_permutation_p"], 2 / 6)
        self.assertAlmostEqual(result["add_one_two_sided_permutation_p"], 3 / 7)


if __name__ == "__main__":
    unittest.main()
