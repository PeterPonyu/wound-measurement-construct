"""RT06 whole-raw pipeline contracts; synthetic fixtures are not biological data."""
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_full_pipeline_depth.py"
SPEC = importlib.util.spec_from_file_location("full_pipeline_depth_audit", SCRIPT)
DEPTH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DEPTH)


def cell_frame(scores, qc, labels, patient="P1", arm="DFU-healer"):
    n = len(scores)
    return pd.DataFrame({
        "gsm": ["GSM1"] * n, "arm": [arm] * n, "patient_id": [patient] * n,
        "raw_cell_id": [f"GSM1:BC{i}" for i in range(n)],
        "raw_row_0": np.arange(n), "qc_pass": qc, "celltype": labels,
        "topic_0": scores, "score_available": np.isfinite(scores),
        "marker_top_two_margin": np.ones(n), "detected_marker_genes": np.ones(n),
        "library_size": np.full(n, 300), "detected_genes": np.full(n, 220),
    })


class FullPipelineDepthAudit(unittest.TestCase):
    def test_protocol_prespecifies_complete_grid_and_does_not_offer_threshold_tuning(self):
        protocol = json.loads(DEPTH.DEFAULT_PROTOCOL.read_text())
        DEPTH.validate_protocol(protocol)
        self.assertEqual(protocol["cohort"]["expected_pre_qc_cells"], 81602)
        self.assertEqual(protocol["cohort"]["expected_baseline_qc_cells"], 76461)
        protocol["perturbation"]["seeds"] = [20260925]
        with self.assertRaises(ValueError):
            DEPTH.validate_protocol(protocol)

    def test_streaming_reader_preserves_all_pre_qc_cells_and_source_order(self):
        text = b',BC3,BC1,BC2\nG2,0,2,0\nG1,5,0,0\nMT-A,0,1,0\n'
        counts, genes, barcodes, full_columns = DEPTH.read_count_member(
            gzip.compress(text), chunk_rows=1)
        np.testing.assert_array_equal(genes, ["G2", "G1", "MT-A"])
        np.testing.assert_array_equal(barcodes, ["BC3", "BC1", "BC2"])
        np.testing.assert_array_equal(counts.toarray(), [[0, 5, 0], [2, 0, 1], [0, 0, 0]])
        self.assertEqual(full_columns, 3)
        self.assertEqual(counts.dtype, np.dtype("int32"))
        # Smoke subset must disclose the original, larger column count.
        subset, _, _, original_n = DEPTH.read_count_member(text, compressed=False, max_cells=1)
        self.assertEqual((subset.shape[0], original_n), (1, 3))

    def test_implicit_r_row_index_does_not_drop_first_barcode(self):
        text = b'BC3,BC1,BC2\nG2,0,2,0\nG1,5,0,0\nMT-A,0,1,0\n'
        counts, genes, barcodes, n = DEPTH.read_count_member(text, compressed=False, chunk_rows=1)
        np.testing.assert_array_equal(barcodes, ["BC3", "BC1", "BC2"])
        np.testing.assert_array_equal(counts.toarray(), [[0, 5, 0], [2, 0, 1], [0, 0, 0]])
        subset, _, subset_barcodes, n = DEPTH.read_count_member(text, compressed=False, max_cells=1)
        np.testing.assert_array_equal(subset.toarray(), [[0, 5, 0]])
        np.testing.assert_array_equal(subset_barcodes, ["BC3"])
        self.assertEqual(n, 3)

    def test_source_validation_rejects_fractional_negative_duplicate_and_overflow(self):
        bad_sources = [b',A\nG,0.1\n', b',A\nG,-1\n', b',A\nG,2147483648\n',
                       b',A,A\nG,1,2\n', b',A\nG,1\nG,2\n']
        for data in bad_sources:
            with self.subTest(data=data), self.assertRaises(ValueError):
                DEPTH.read_count_member(data, compressed=False, chunk_rows=1)

    def test_binomial_thinning_is_integer_bounded_reproducible_and_sample_specific(self):
        counts = sp.csr_matrix(np.full((100, 4), 12, dtype=np.int32))
        original = counts.copy()
        first = DEPTH.thin_counts(counts, 0.5, 17, "GSM100")
        again = DEPTH.thin_counts(counts, 0.5, 17, "GSM100")
        other_seed = DEPTH.thin_counts(counts, 0.5, 18, "GSM100")
        other_sample = DEPTH.thin_counts(counts, 0.5, 17, "GSM101")
        np.testing.assert_array_equal(first.toarray(), again.toarray())
        np.testing.assert_array_equal(counts.toarray(), original.toarray())
        self.assertFalse(np.array_equal(first.toarray(), other_seed.toarray()))
        self.assertFalse(np.array_equal(first.toarray(), other_sample.toarray()))
        self.assertTrue((first.toarray() <= counts.toarray()).all())
        self.assertGreater(first.sum() / counts.sum(), 0.45)
        self.assertLess(first.sum() / counts.sum(), 0.55)
        np.testing.assert_array_equal(DEPTH.thin_counts(counts, 1, 1, "GSM100").toarray(), counts.toarray())
        self.assertEqual(DEPTH.thin_counts(counts, 0, 1, "GSM100").nnz, 0)
        with self.assertRaises(TypeError):
            DEPTH.thin_counts(counts.astype(float), .5, 1, "GSM100")

    def test_qc_recomputed_on_all_genes_can_lose_and_admit_original_cells(self):
        genes = np.array([f"G{i}" for i in range(200)] + ["MT-A"])
        original = np.ones((3, 201), dtype=np.int32)
        original[:, -1] = [40, 80, 0]
        original[2] = 0
        perturbed = original.copy()
        perturbed[0, 0] = 0
        perturbed[0, 1] = 0  # drop below 200 detected genes
        perturbed[1, -1] = 10  # mitochondrial fraction now passes
        before, *_ = DEPTH.qc_metrics(sp.csr_matrix(original), genes)
        after, *_ = DEPTH.qc_metrics(sp.csr_matrix(perturbed), genes)
        np.testing.assert_array_equal(before, [True, False, False])
        np.testing.assert_array_equal(after, [False, True, False])

    def test_sample_missing_genes_are_zero_filled_not_minus_one_indexed(self):
        counts = sp.csr_matrix(np.array([[2, 7], [3, 9]], dtype=np.int32))
        actual, coverage = DEPTH.align_panel(counts, ["A", "B"], ["B", "missing", "A"])
        np.testing.assert_array_equal(actual.toarray(), [[7, 0, 2], [9, 0, 3]])
        self.assertEqual(coverage, 2)

    def test_gate_uses_frozen_reference_with_zero_fill_and_preserves_low_information(self):
        reference = {"lineages": ["fibroblast", "other"],
                     "marker_genes": ["A", "B", "C", "D", "E", "F"],
                     "markers_by_lineage": {"fibroblast": ["A", "B", "C"], "other": ["D", "E", "F"]},
                     "means": {"fibroblast": np.zeros(3), "other": np.zeros(3)},
                     "sds": {"fibroblast": np.ones(3), "other": np.ones(3)}}
        counts = sp.csr_matrix(np.array([[5, 4, 3, 0, 0, 0, 1],
                                        [0, 0, 0, 5, 4, 3, 1],
                                        [0, 0, 0, 0, 0, 0, 9]], dtype=np.int32))
        labels, margin, detected = DEPTH.frozen_gate(counts, ["A", "B", "C", "D", "E", "F", "X"], reference)
        np.testing.assert_array_equal(labels[:2], ["fibroblast", "other"])
        self.assertEqual((detected[2], margin[2]), (0, 0))
        np.testing.assert_array_equal(reference["means"]["fibroblast"], np.zeros(3))
        before, *_ = DEPTH.frozen_gate(counts[:1], ["A", "B", "C", "D", "E", "F", "X"], reference)
        np.testing.assert_array_equal(before, labels[:1])

    def test_projection_is_softmax_mu_frozen_deterministic_and_zero_library_is_missing(self):
        torch.manual_seed(2)
        model = DEPTH.TopicModel(3, num_topics=2, hidden_dim=8).eval()
        counts = sp.csr_matrix(np.array([[1, 2, 3], [0, 0, 0], [4, 2, 0]], dtype=np.int32))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        theta, nonzero = DEPTH.project_panel(counts, model, batch_size=2)
        repeated, _ = DEPTH.project_panel(counts, model, batch_size=1)
        np.testing.assert_allclose(theta, repeated, equal_nan=True, atol=1e-7)
        with torch.no_grad():
            expected = model(torch.from_numpy(DEPTH.library_normalize(counts[nonzero]).toarray()), deterministic=True)["theta"].numpy()
        np.testing.assert_allclose(theta[nonzero], expected, atol=1e-7)
        self.assertTrue(np.isnan(theta[1]).all())
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_population_branches_keep_score_qc_gate_estimands_separate(self):
        baseline = cell_frame([.1, .2, .3, .4], [True, True, False, False],
                              ["fibroblast", "other", "other", "other"])
        current = cell_frame([.2, .4, .6, .8], [False, True, True, False],
                             ["fibroblast", "fibroblast", "other", "other"])
        masks = DEPTH.population_masks(baseline, current)
        np.testing.assert_array_equal(masks["full_pipeline"][0], [False, True, True, False])
        np.testing.assert_array_equal(masks["original_qc_fixed_gate"][0], [True, True, False, False])
        np.testing.assert_array_equal(masks["qc_intersection_fixed_gate"][0], [False, True, False, False])
        self.assertEqual(masks["qc_intersection_fixed_gate"][1][1], "other")
        self.assertEqual(masks["qc_intersection_regated"][1][1], "fibroblast")
        record, _ = DEPTH.transition_record(baseline, current, "example", .5, 1)
        self.assertEqual((record["lost_qc"], record["gained_qc"], record["common_qc_fibroblast_entry"]), (1, 1, 1))

    def test_all_cell_and_fibroblast_patient_weights_and_empty_specimen(self):
        first = cell_frame([.1, .9], [True, True], ["fibroblast", "other"])
        second = cell_frame([.3, .5, .7], [True] * 3, ["fibroblast"] * 3)
        second["gsm"] = "GSM2"
        empty = cell_frame([.9], [False], ["other"])
        empty["gsm"] = "GSM3"
        rows = []
        for frame in (first, second, empty):
            rows.extend(DEPTH.specimen_readouts(frame, frame, "baseline", 1, -1, .4))
        patients = DEPTH.aggregate_patients(pd.DataFrame(rows))
        patient = patients.loc[patients.population.eq("full_pipeline")].iloc[0]
        self.assertEqual((patient.n_specimens, patient.n_empty_specimens, patient.n_all, patient.n_fibroblast), (3, 1, 5, 4))
        self.assertAlmostEqual(patient.all_cell_topic0_mean, .5)
        self.assertAlmostEqual(patient.fibroblast_topic0_mean, .4)
        self.assertAlmostEqual(patient.fibroblast_high_state_fraction, .5)
        self.assertAlmostEqual(patient.fibroblast_fraction, .8)

    def test_zero_panel_library_not_silently_dropped_from_denominators(self):
        frame = cell_frame([.1, np.nan], [True, True], ["fibroblast", "fibroblast"])
        row = DEPTH.readout_row(frame, np.ones(2, bool), frame.celltype.to_numpy(), .2)
        self.assertEqual((row["n_all"], row["n_fibroblast"], row["n_missing_score"]), (2, 2, 1))
        self.assertTrue(np.isnan(row["all_cell_topic0_mean"]))
        self.assertTrue(np.isnan(row["fibroblast_topic0_mean"]))
        self.assertEqual(row["fibroblast_fraction"], 1)

    def test_outcome_test_excludes_healthy_and_has_330_allocations_not_cells(self):
        rows = []
        for idx, arm in enumerate(["DFU-healer"] * 7 + ["DFU-nonhealer"] * 4 + ["Non-diabetic"]):
            record = dict(run="baseline", rate=1., seed=-1, population="full_pipeline",
                          patient_id=f"P{idx:02d}", arm=arm)
            record.update({readout: idx / 20 for readout in DEPTH.READOUTS})
            rows.append(record)
        patients = pd.DataFrame(rows)
        contrasts = DEPTH.outcome_contrasts(patients)
        self.assertTrue(contrasts.allocations.eq(330).all())
        self.assertTrue(contrasts.n_healed.eq(7).all())
        self.assertTrue(contrasts.n_nonhealed.eq(4).all())
        self.assertTrue(contrasts.healthy_patients_excluded.eq(1).all())
        changed = patients.copy()
        changed.loc[changed.arm.eq("Non-diabetic"), list(DEPTH.READOUTS)] = -10000
        np.testing.assert_array_equal(contrasts.exact_p, DEPTH.outcome_contrasts(changed).exact_p)
        patients.loc[0, "all_cell_topic0_mean"] = np.nan
        missing = DEPTH.outcome_contrasts(patients)
        self.assertEqual(missing.iloc[0].status, "not_estimable")
        self.assertEqual(missing.iloc[0].missing_patients, "P00")

    def test_pair_stability_reports_selection_and_missing_without_truth_claim(self):
        original = cell_frame([.1, .3, .5], [True] * 3, ["fibroblast"] * 3)
        current = cell_frame([.2, .4, np.nan], [True, True, False], ["fibroblast"] * 3)
        result = DEPTH.paired_stability(original, current, np.ones(3, bool), .35)
        self.assertEqual((result["n_selected"], result["n_paired"], result["n_missing_pairs"]), (3, 2, 1))
        self.assertAlmostEqual(result["mean_absolute_change"], .1)
        self.assertAlmostEqual(result["historical_threshold_crossing_fraction"], .5)

    def test_existing_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run"
            DEPTH.fresh_directory(path)
            with self.assertRaises(FileExistsError):
                DEPTH.fresh_directory(path)

    def test_requested_model_is_not_claimed_as_runtime_verification(self):
        evidence = DEPTH.runtime_provenance(None, "gpt-6-astra")
        self.assertEqual(evidence["requested_model"], "gpt-6-astra")
        self.assertIsNone(evidence["runtime_reported_model"])
        self.assertFalse(evidence["same_exact_model_verified"])


if __name__ == "__main__":
    unittest.main()
