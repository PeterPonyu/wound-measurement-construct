#!/usr/bin/env python3
"""
Topic Simplex Engine (Ported & Modularized from PanODE-Topic & Topic-Core).
Strictly independent module: Implements Dirichlet Simplex Decoder & Topic Representation.

Mathematical Formulation:
- Input expression: x in R^G (HVGs)
- Encoder: q_phi(z | x) -> mu, log_sigma in R^K (pre-softmax space)
- Reparameterization & Simplex projection:
    z ~ N(mu, diag(sigma^2))
    theta = Softmax(z) in Delta^{K-1}
- Decoder:
    beta in Delta^{G-1} per topic (Topic-Gene matrix)
    x_hat = theta @ beta
- Loss:
    L = Recon_Loss(x, x_hat) + kl_weight * KL(q_phi(z|x) || p(z))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class TopicEncoder(nn.Module):
    """MLP encoder predicting pre-softmax topic coordinates."""
    def __init__(self, input_dim: int, hidden_dim: int, num_topics: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, num_topics)
        self.fc_logvar = nn.Linear(hidden_dim // 2, num_topics)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class SimplexDecoder(nn.Module):
    """
    Decodes topic mixture weights theta in Delta^{K-1} into gene expression.
    Maintains a normalized topic-gene matrix beta in Delta^{G-1} per topic.
    """
    def __init__(self, num_topics: int, output_dim: int):
        super().__init__()
        self.num_topics = num_topics
        self.output_dim = output_dim
        # Unconstrained topic embeddings transformed via softmax over genes
        self.beta_raw = nn.Parameter(torch.randn(num_topics, output_dim) * 0.02)

    @property
    def beta(self) -> torch.Tensor:
        """Returns topic-gene matrix with each row normalized to sum to 1."""
        return F.softmax(self.beta_raw, dim=-1)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            theta: [batch_size, num_topics] on simplex
        Returns:
            x_recon: [batch_size, output_dim] reconstructed gene proportions
        """
        return torch.matmul(theta, self.beta)

class TopicSimplexModel(nn.Module):
    """Complete, self-contained Topic Simplex representation model."""
    def __init__(self, input_dim: int, num_topics: int = 10, hidden_dim: int = 256, kl_weight: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.num_topics = num_topics
        self.kl_weight = kl_weight
        
        self.encoder = TopicEncoder(input_dim, hidden_dim, num_topics)
        self.decoder = SimplexDecoder(num_topics, input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=-1) # Project onto probability simplex
        x_recon = self.decoder(theta)
        
        # Dirichlet/Normal prior KL: standard Gaussian in pre-softmax space
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        
        # Multinomial log-likelihood / Cross-entropy reconstruction loss
        eps = 1e-10
        recon_loss = -torch.sum(x * torch.log(x_recon + eps), dim=-1).mean()
        total_loss = recon_loss + self.kl_weight * kl_div
        
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_div": kl_div,
            "theta": theta,
            "z": z,
            "mu": mu,
            "x_recon": x_recon
        }

if __name__ == "__main__":
    print("=" * 60)
    print("Self-Testing Topic Simplex Engine...")
    torch.manual_seed(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    batch_size, n_genes, k_topics = 64, 500, 8
    dummy_x = torch.poisson(torch.rand(batch_size, n_genes) * 3.0).to(device)
    # Normalize per cell (library size normalization)
    dummy_x_norm = dummy_x / (dummy_x.sum(dim=-1, keepdim=True) + 1e-8)
    
    model = TopicSimplexModel(input_dim=n_genes, num_topics=k_topics).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Run 5 optimization steps
    model.train()
    for step in range(5):
        optimizer.zero_grad()
        out = model(dummy_x_norm)
        out["loss"].backward()
        optimizer.step()
        
    theta_sample = out["theta"].detach().cpu()
    print(f"Loss after 5 steps: {out['loss'].item():.4f}")
    print(f"Theta shape: {theta_sample.shape}")
    print(f"Theta row sums (must be 1.0 on simplex): {theta_sample.sum(dim=-1)[:5].numpy()}")
    assert torch.allclose(theta_sample.sum(dim=-1), torch.ones(batch_size), atol=1e-5), "Simplex constraint failed!"
    print("Topic Simplex Engine test PASSED!")
    print("=" * 60)
