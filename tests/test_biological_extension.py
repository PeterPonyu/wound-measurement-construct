"""Analytic checks for the added estimands, independent of fitted outcomes."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
spec = importlib.util.spec_from_file_location('extension', Path(__file__).resolve().parents[1] / 'scripts/expand_biological_evidence.py')
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)

class BiologicalExtensionTests(unittest.TestCase):

    def test_pooling_respects_each_denominator(self):
        rows = pd.DataFrame([dict(patient='A', arm='healed', n_all=100, n_fib=10, fibroblast_mean=0.2, high_state_fraction=0.25, fibroblast_fraction=0.1), dict(patient='A', arm='healed', n_all=100, n_fib=30, fibroblast_mean=0.6, high_state_fraction=0.75, fibroblast_fraction=0.3)])
        pooled = extension.patient_rows(rows).iloc[0]
        self.assertAlmostEqual(pooled.fibroblast_mean, 0.5)
        self.assertAlmostEqual(pooled.high_state_fraction, 0.625)
        self.assertAlmostEqual(pooled.fibroblast_fraction, 0.2)
        self.assertAlmostEqual(extension.patient_rows(rows, equal=True).iloc[0].fibroblast_mean, 0.4)

    def test_exhaustive_probability_includes_extreme_allocations_and_ties(self):
        x = np.arange(11, dtype=float)
        labels = np.arange(11) >= 4
        difference, p, n = extension.exact_contrast(x, labels)
        self.assertEqual(n, 330)
        self.assertAlmostEqual(difference, 5.5)
        self.assertAlmostEqual(p, 2 / 330)
        self.assertEqual(extension.exact_contrast(np.ones(11), labels)[1], 1)
        self.assertAlmostEqual(extension.exact_contrast(7 + 3 * x, labels)[1], p)
if __name__ == '__main__':
    unittest.main()
