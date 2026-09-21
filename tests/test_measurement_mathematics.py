"""Numerical checks of representation, mixture and patient-level estimands."""
import unittest

import numpy as np
from scipy.stats import t
import torch

from wound_models.topic_model import TopicModel
from scripts.audit_measurement_math import state_summary, exhaustive_contrast, equivalence_boundary


class MeasurementMathematics(unittest.TestCase):
    def test_simplex_and_gaussian_kl(self):
        torch.manual_seed(71)
        model = TopicModel(input_dim=7, num_topics=3, hidden_dim=8).double().eval()
        x = torch.rand(5, 7, dtype=torch.float64)
        x /= x.sum(1, keepdim=True)
        first, second = model(x), model(x)
        self.assertTrue(torch.equal(first['theta'], second['theta']))
        torch.testing.assert_close(first['theta'].sum(1), torch.ones(5, dtype=torch.float64))
        torch.testing.assert_close(first['x_recon'].sum(1), torch.ones(5, dtype=torch.float64))
        mu, logvar = model.encoder(x)
        q = torch.distributions.Normal(mu, torch.exp(logvar / 2))
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
        expected = torch.distributions.kl_divergence(q, p).sum(1).mean()
        torch.testing.assert_close(first['kl_div'], expected)
        torch.testing.assert_close(first['loss'], first['recon_loss'] + 0.01 * expected)

    def test_simplex_does_not_identify_common_coordinate_offset(self):
        z = torch.tensor([[1., 2., -1.]], dtype=torch.float64)
        torch.testing.assert_close(torch.softmax(z, -1), torch.softmax(z + 30, -1))
        self.assertGreater(torch.linalg.vector_norm((z + 30) - z).item(), 50)

    def test_state_identity_including_empty_bins(self):
        for values in ([0.02, 0.04], [0.4, 0.6], [0.01, 0.05, 0.3, 0.7]):
            row = state_summary(values, 0.184, 0.03, 0.5)
            self.assertLess(row['identity_error'], 1e-14)
            self.assertLess(row['residual_identity_error'], 1e-14)
        # Equal fractions can accompany very different mean loadings.
        a = state_summary([0.02, 0.3], 0.184, 0.03, 0.5)
        b = state_summary([0.1, 0.7], 0.184, 0.03, 0.5)
        self.assertEqual(a['high_state_fraction'], b['high_state_fraction'])
        self.assertGreater(b['mean_loading'] - a['mean_loading'], 0.2)

    def test_exhaustive_probability_counts_observed_allocation(self):
        result = exhaustive_contrast([0, 1, 2, 3], [False, False, True, True])
        self.assertEqual(result['allocations'], 6)
        self.assertEqual(result['extreme_allocations'], 2)
        self.assertAlmostEqual(result['exact_probability'], 1 / 3)
        self.assertAlmostEqual(result['recorded_add_one_probability'], 3 / 7)

    def test_tost_boundary_matches_one_sided_probabilities(self):
        a, b = np.array([0.02, 0.04, 0.12, 0.07]), np.array([0.01, 0.03, 0.06])
        result = equivalence_boundary(a, b)
        delta, se, df = a.mean() - b.mean(), result['standard_error'], result['welch_df']
        margin = result['equivalence_margin_infimum']
        probabilities = [t.sf((delta + margin) / se, df), t.cdf((delta - margin) / se, df)]
        self.assertAlmostEqual(max(probabilities), 0.05)
        self.assertAlmostEqual(margin, max(abs(v) for v in result['two_sided_90_interval']))


if __name__ == '__main__':
    unittest.main()
