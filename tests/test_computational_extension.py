"""Analytic invariants for patient measures and endpoint-only predictors."""
import unittest
import numpy as np
from scripts.strengthen_computational_evidence import allocation_weights, ecdf_contrast

class ComputationalExtensionTests(unittest.TestCase):

    def test_cdf_test_counts_patients_and_preserves_within_patient_replication(self):
        values = [np.array([0.0, 0.2]), np.array([0.1, 0.3]), np.array([0.7, 0.9]), np.array([0.8, 1.0])]
        labels = np.array([True, True, False, False])
        result = ecdf_contrast(values, labels)
        self.assertEqual(len(result[2]), 6)
        self.assertAlmostEqual(result[3], 1.0)
        self.assertAlmostEqual(result[4], 2 / 6)
        repeated = [np.repeat(v, i + 2) for i, v in enumerate(values)]
        for a, b in zip(result, ecdf_contrast(repeated, labels)):
            np.testing.assert_allclose(a, b)
        self.assertEqual(ecdf_contrast([np.array([0.5])] * 4, labels)[4], 1.0)

    def test_allocation_matrix_is_exhaustive_and_gives_equal_patient_mass(self):
        w = allocation_weights(11, 7)
        self.assertEqual(w.shape, (330, 11))
        np.testing.assert_allclose(w.sum(1), 0, atol=1e-15)
        self.assertTrue(np.all((w > 0).sum(1) == 7))
        self.assertEqual(len(np.unique(w, axis=0)), 330)
if __name__ == '__main__':
    unittest.main()
