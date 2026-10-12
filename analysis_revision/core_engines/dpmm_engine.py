#!/usr/bin/env python3
"""
DPMM Mixture Engine (Ported & Modularized from PanODE-DPMM & DPMM-Core).
Strictly independent module: Implements non-parametric mixture allocation & occupancy tracking.

Mathematical Formulation:
- Truncated stick-breaking construction of Dirichlet Process (DP):
    v_k ~ Beta(1, alpha_dp),  k = 1, ..., K_max
    pi_1 = v_1
    pi_k = v_k * prod_{j < k} (1 - v_j)
- Cell assignment likelihood:
    gamma_{ik} = P(z_i = k | x_i) = Softmax( log pi_k + log N(x_i | mu_k, Sigma_k) )
- Effective Occupancy metric (as defined in DPMM-Core):
    K_{>1%} = sum_{k=1}^{K_max} I( (1/N) * sum_i gamma_{ik} > 0.01 )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class DPMMGaussianMixture(nn.Module):
    """
    Variational Gaussian Mixture with Truncated Dirichlet Process Prior.
    Learns mixture component means mu_k, log-variances, and mixture responsibilities.
    """
    def __init__(self, latent_dim: int, max_components: int = 25, alpha_dp: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_components = max_components
        self.alpha_dp = alpha_dp
        
        # Component parameters in latent space
        self.mu = nn.Parameter(torch.randn(max_components, latent_dim) * 0.5)
        self.log_var = nn.Parameter(torch.zeros(max_components, latent_dim))
        
        # Variational stick-breaking parameters
        self.stick_v_logit = nn.Parameter(torch.zeros(max_components - 1))

    def compute_mixture_weights(self) -> torch.Tensor:
        """
        Computes mixture prior weights pi_k via stick-breaking construction.
        Returns [max_components] summing to 1.
        """
        v = torch.sigmoid(self.stick_v_logit) # [K-1]
        
        pi = []
        remaining_stick = 1.0
        for k in range(self.max_components - 1):
            weight = v[k] * remaining_stick
            pi.append(weight)
            remaining_stick = remaining_stick * (1.0 - v[k])
        pi.append(remaining_stick)
        
        return torch.stack(pi)

    def compute_responsibilities(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute soft responsibilities gamma_{ik} = P(cluster = k | z_i).
        Args:
            z: [batch_size, latent_dim]
        Returns:
            gamma: [batch_size, max_components]
            log_likelihood: [batch_size]
        """
        pi = self.compute_mixture_weights() # [K]
        log_pi = torch.log(pi + 1e-12) # [K]
        
        # Compute log N(z | mu_k, Sigma_k) for all components
        # z: [B, 1, D], mu: [1, K, D], var: [1, K, D]
        z_exp = z.unsqueeze(1)
        mu_exp = self.mu.unsqueeze(0)
        var_exp = torch.exp(self.log_var).unsqueeze(0)
        
        diff = z_exp - mu_exp # [B, K, D]
        quad = -0.5 * torch.sum(diff.pow(2) / (var_exp + 1e-8), dim=-1) # [B, K]
        log_det = -0.5 * torch.sum(self.log_var, dim=-1).unsqueeze(0) # [1, K]
        const = -0.5 * self.latent_dim * 1.837877 # log(2*pi)
        
        log_prob = quad + log_det + const # [B, K]
        log_resp_unnorm = log_prob + log_pi.unsqueeze(0) # [B, K]
        
        # Log-Sum-Exp over components for total marginal likelihood
        log_marginal = torch.logsumexp(log_resp_unnorm, dim=-1) # [B]
        gamma = torch.softmax(log_resp_unnorm, dim=-1) # [B, K]
        
        return gamma, log_marginal

    def compute_occupancy(self, gamma: torch.Tensor, threshold: float = 0.01) -> int:
        """
        Computes K_{>1%}: number of mixture components carrying > 1% of total batch mass.
        """
        mean_mass_per_component = gamma.mean(dim=0) # [K]
        k_active = (mean_mass_per_component > threshold).sum().item()
        return int(k_active)

if __name__ == "__main__":
    print("=" * 60)
    print("Self-Testing DPMM Mixture Engine...")
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    latent_dim = 8
    max_k = 20
    model = DPMMGaussianMixture(latent_dim=latent_dim, max_components=max_k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    # Generate synthetic 3-modal data in latent space
    z1 = torch.randn(50, latent_dim, device=device) + torch.tensor([3.0] * latent_dim, device=device)
    z2 = torch.randn(50, latent_dim, device=device) - torch.tensor([3.0] * latent_dim, device=device)
    z3 = torch.randn(50, latent_dim, device=device)
    z_synth = torch.cat([z1, z2, z3], dim=0)
    
    # Train mixture parameters for 25 steps
    model.train()
    for step in range(25):
        optimizer.zero_grad()
        gamma, log_lik = model.compute_responsibilities(z_synth)
        loss = -log_lik.mean() # Maximize marginal likelihood
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        gamma, _ = model.compute_responsibilities(z_synth)
        k_active = model.compute_occupancy(gamma, threshold=0.01)
        pi_weights = model.compute_mixture_weights()
        
    print(f"Total components evaluated: {max_k}")
    print(f"K_(>1%) Active Occupancy discovered: {k_active}")
    print(f"Top 5 mixture weights: {pi_weights[:5].cpu().numpy().round(3)}")
    assert gamma.sum(dim=-1).allclose(torch.ones(150, device=device)), "Responsibilities must sum to 1!"
    assert k_active >= 2, "DPMM should discover at least 2 active clusters!"
    print("DPMM Mixture Engine test PASSED!")
    print("=" * 60)
