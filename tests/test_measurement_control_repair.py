"""Behavioral checks for the repaired measurement controls."""
import unittest
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/repair_measurement_controls.py"
_SPEC = importlib.util.spec_from_file_location("measurement_control_repair", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
apply_marker_reference = _MODULE.apply_marker_reference
dfu_only_contrast = _MODULE.dfu_only_contrast
exact_contrast = _MODULE.exact_contrast
fit_marker_reference = _MODULE.fit_marker_reference
shuffle_columns_exact = _MODULE.shuffle_columns_exact
verify_column_marginals = _MODULE.verify_column_marginals
deterministic_project = _MODULE.deterministic_project
marker_log_expression = _MODULE.marker_log_expression
patient_readouts = _MODULE.patient_readouts
from wound_models.celltype_markers import score_cell_types
from wound_models.topic_model import TopicModel


class MeasurementControlRepair(unittest.TestCase):
    def test_raw_column_shuffle_preserves_each_integer_marginal(self):
        source = np.array([
            [1, 0, 10],
            [2, 1, 0],
            [3, 2, 0],
            [4, 3, 0],
        ], dtype=np.int32)
        shuffled = shuffle_columns_exact(source, seed=17)
        check = verify_column_marginals(source, shuffled)
        self.assertTrue(check["every_column_exact"])
        self.assertEqual(int(np.count_nonzero(source)), int(np.count_nonzero(shuffled)))
        self.assertFalse(np.array_equal(source.sum(axis=1), shuffled.sum(axis=1)))
        # Correlated columns with identifiable rows must receive distinct permutations.
        correlated = np.repeat(np.arange(30, dtype=np.int32)[:, None], 3, axis=1)
        independent = shuffle_columns_exact(correlated, seed=17)
        self.assertFalse(np.array_equal(independent[:, 0], independent[:, 1]))

    def test_decoder_proportions_and_corrupted_marginals_are_rejected(self):
        with self.assertRaises(TypeError):
            shuffle_columns_exact(np.full((4, 3), 1 / 3, dtype=np.float32))
        counts = np.arange(12, dtype=np.int32).reshape(4, 3)
        changed = counts.copy()
        changed[0, 0] += 1
        with self.assertRaises(AssertionError):
            verify_column_marginals(counts, changed)

    def test_frozen_marker_reference_is_invariant_to_later_cells(self):
        genes = ["a", "b", "c", "d", "e", "f"]
        panel = {"fibroblast": ["a", "b", "c"],
                 "keratinocyte": ["d", "e", "f"]}
        expression = np.array([
            [5, 4, 3, 0, 0, 0],
            [4, 5, 3, 0, 0, 0],
            [0, 0, 0, 5, 4, 3],
            [0, 0, 0, 4, 5, 3],
        ], dtype=np.float32)
        reference = fit_marker_reference(expression, genes, panel=panel)
        before, _ = apply_marker_reference(expression, reference)
        appended = np.vstack([expression, [[100, 0, 0, 0, 0, 0]]])
        after, _ = apply_marker_reference(appended, reference)
        np.testing.assert_array_equal(before, after[: len(before)])
        self.assertEqual(reference["n_cells"], 4)

    def test_reference_fit_reproduces_existing_count_based_marker_scorer(self):
        genes = np.array(["a", "b", "c", "d", "e", "f", "other"])
        panel = {"fibroblast": ["a", "b", "c"], "keratinocyte": ["d", "e", "f"]}
        counts = sp.csr_matrix(np.random.default_rng(91).poisson(2, size=(60, 7)).astype(np.int32))
        expected, expected_scores, _ = score_cell_types(counts, genes, panel=panel)
        expression, markers = marker_log_expression(counts, genes, panel)
        reference = fit_marker_reference(expression, markers, panel)
        labels, scores = apply_marker_reference(expression, reference)
        np.testing.assert_array_equal(labels, expected)
        np.testing.assert_allclose(scores, expected_scores, atol=1e-6, rtol=1e-6)

    def test_deterministic_projection_uses_frozen_softmax_mu(self):
        torch.manual_seed(3)

        class RecordingModel(TopicModel):
            requested_modes = []

            def forward(self, x, deterministic=None):
                self.requested_modes.append(deterministic)
                return super().forward(x, deterministic=deterministic)

        model = RecordingModel(input_dim=3, num_topics=2, hidden_dim=8).eval()
        counts = sp.csr_matrix(np.array([
            [3, 1, 0], [0, 2, 4], [1, 1, 1], [6, 0, 2],
        ], dtype=np.int32))
        first, first_mu = deterministic_project(counts, np.arange(3), model, "cpu", batch_size=2)
        second, second_mu = deterministic_project(counts, np.arange(3), model, "cpu", batch_size=3)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_mu, second_mu)
        np.testing.assert_allclose(first.sum(axis=1), 1.0, atol=1e-6)
        self.assertEqual(model.requested_modes, [True, True, True, True])
        np.testing.assert_allclose(first, torch.softmax(torch.from_numpy(first_mu), dim=1).numpy(), atol=1e-7)

    def test_dfu_only_allocation_excludes_healthy_controls(self):
        table = pd.DataFrame({
            "patient_id": ["P1", "P2", "P3", "P4", "P5"],
            "arm": ["DFU-healer", "DFU-healer", "DFU-nonhealer",
                    "DFU-nonhealer", "Non-diabetic"],
            "fibroblast_mean": [0.1, 0.4, 0.2, 0.3, 0.99],
        })
        result, null = dfu_only_contrast(table, "fibroblast_mean")
        self.assertEqual((result["n_healed"], result["n_nonhealed"]), (2, 2))
        self.assertEqual(result["allocations"], 6)
        self.assertEqual(result["healthy_rows_excluded_before_allocation"], 1)
        self.assertEqual(len(null), 6)

    def test_empty_fibroblast_specimen_is_retained_in_patient_denominators(self):
        samples = pd.DataFrame({
            "patient_id": ["P1", "P1", "P2"], "arm": ["DFU-healer"] * 3,
            "n_fibroblast": [0, 10, 0], "n_all": [20, 20, 20],
            "fibroblast_mean": [np.nan, 0.3, np.nan],
            "high_state_fraction": [np.nan, 0.4, np.nan],
            "fibroblast_fraction": [0.0, 0.5, 0.0],
        })
        patients = patient_readouts(samples).set_index("patient_id")
        self.assertEqual(patients.loc["P1", "specimens"], 2)
        self.assertEqual(patients.loc["P1", "n_all"], 40)
        self.assertEqual(patients.loc["P1", "fibroblast_mean"], 0.3)
        self.assertEqual(patients.loc["P1", "fibroblast_fraction"], 0.25)
        self.assertTrue(patients.loc["P2", "zero_fibroblast"])
        self.assertTrue(np.isnan(patients.loc["P2", "fibroblast_mean"]))

    def test_exact_contrast_counts_ties_without_add_one(self):
        observed, p_value, allocations, null = exact_contrast(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([False, False, True, True]),
        )
        self.assertAlmostEqual(observed, 2.0)
        self.assertEqual(allocations, 6)
        self.assertAlmostEqual(p_value, 1 / 3)
        self.assertEqual(np.count_nonzero(np.isclose(np.abs(null), 2.0)), 2)


if __name__ == "__main__":
    unittest.main()
