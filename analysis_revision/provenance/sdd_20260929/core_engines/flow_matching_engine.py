#!/usr/bin/env python3
"""
Conditional flow matching with linear endpoint paths and fixed-step integration.

Mathematical Formulation:
- Source distribution: z_0 ~ N(0, I) or initial state z_early
- Target distribution: an observed endpoint distribution
- Linear conditional probability path on the unit interval:
    z_t = (1 - t) * z_0 + t * z_1,  t in [0, 1]
    Target velocity: u_t(z_t | z_0, z_1) = z_1 - z_0
- Flow matching objective:
    L_CFM(theta) = E_{t ~ U(0,1), z_0, z_1} [ || v_theta(z_t, t) - (z_1 - z_0) ||_2^2 ]
- Model trajectory integration:
    dz/dt = v_theta(z, t), solved via Runge-Kutta 4th Order.
On a segment [a, b], target velocity is (z_1-z_0)/(b-a). The caller
specifies the endpoint coupling; independent draws are not an optimal
transport plan. Reconstructed marginals do not identify cell ancestry.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal continuous time embedding for t in [0, 1]."""
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 2:
            t = t.squeeze(-1)

        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.embed_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class LatentFlowField(nn.Module):
    """MLP parameterizing the dynamic velocity field v_theta(z_t, t)."""
    def __init__(self, latent_dim: int, hidden_dim: int = 128, time_embed_dim: int = 32):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent coordinates [batch_size, latent_dim]
            t: Continuous time scalar or [batch_size] in [0, 1]
        Returns:
            velocity: dz/dt [batch_size, latent_dim]
        """
        if isinstance(t, (int, float)):
            t = torch.full((z.size(0),), t, device=z.device, dtype=z.dtype)
        elif t.dim() == 0:
            t = t.expand(z.size(0))
            
        t_emb = self.time_embed(t)
        inp = torch.cat([z, t_emb], dim=-1)
        return self.net(inp)

class ConditionalLatentFlowField(nn.Module):
    """
    Velocity field v_theta(z, t, c) conditioned on a per-individual context c.

    Context provides additional source-sample information beyond (z, t).
    A shared field can already give different velocities at different z;
    differences between donor-average displacements do not establish that
    conditioning is necessary. Source-only context prevents use of a held-out
    target but does not guarantee transfer to an unseen context.
    """

    def __init__(self, latent_dim: int, context_dim: int, hidden_dim: int = 128,
                 time_embed_dim: int = 32):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim + context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, t, c: torch.Tensor) -> torch.Tensor:
        if isinstance(t, (int, float)):
            t = torch.full((z.size(0),), float(t), device=z.device, dtype=z.dtype)
        elif t.dim() == 0:
            t = t.expand(z.size(0))
        if c.dim() == 1:
            c = c.unsqueeze(0).expand(z.size(0), -1)
        return self.net(torch.cat([z, self.time_embed(t), c], dim=-1))


def compute_conditional_cfm_loss(field: ConditionalLatentFlowField,
                                 z_1: torch.Tensor, z_0: torch.Tensor,
                                 c: torch.Tensor,
                                 t_0: float = 0.0, t_1: float = 1.0,
                                 context_dropout: float = 0.0) -> torch.Tensor:
    """
    Conditional-flow-matching loss for the context-conditioned field.

    context_dropout independently masks each example's entire context vector.
    Training at zero context imposes an additional regression task. It neither
    reproduces a separately trained shared field nor guarantees ODE stability
    or accurate extrapolation to unseen donors.
    """
    batch_size = z_1.size(0)
    span = t_1 - t_0
    if not math.isfinite(span) or span <= 0:
        raise ValueError(f"t_1 must exceed t_0, got t_0={t_0}, t_1={t_1}")
    if not 0 <= context_dropout <= 1:
        raise ValueError('context_dropout must lie in [0, 1]')
    if z_1.ndim != 2 or z_0.shape != z_1.shape or batch_size == 0:
        raise ValueError('Endpoints must be nonempty matching batch-by-coordinate tensors')
    u = torch.rand(batch_size, device=z_1.device)
    z_t = (1.0 - u.view(-1, 1)) * z_0 + u.view(-1, 1) * z_1
    target = (z_1 - z_0) / span
    if c.dim() == 1:
        c = c.unsqueeze(0).expand(batch_size, -1)
    if context_dropout > 0:
        keep = (torch.rand(batch_size, 1, device=z_1.device) >= context_dropout)
        c = c * keep
    return F.mse_loss(field(z_t, t_0 + u * span, c), target)


class FlowMatchingIntegrator:
    """Numerical ODE solver for continuous trajectory integration."""
    def __init__(self, flow_field: LatentFlowField):
        self.flow_field = flow_field

    @torch.no_grad()
    def solve_rk4(self, z0: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
        """
        Runge-Kutta 4th Order numerical integration from t=0 to t=1.
        Returns full trajectory tensor [n_steps + 1, batch_size, latent_dim].
        """
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps <= 0:
            raise ValueError('n_steps must be a positive integer')
        dt = 1.0 / n_steps
        traj = [z0]
        curr_z = z0.clone()

        for step in range(n_steps):
            t = step * dt
            t_half = t + 0.5 * dt
            t_next = (step + 1) * dt

            k1 = self.flow_field(curr_z, t)
            k2 = self.flow_field(curr_z + 0.5 * dt * k1, t_half)
            k3 = self.flow_field(curr_z + 0.5 * dt * k2, t_half)
            k4 = self.flow_field(curr_z + dt * k3, t_next)

            curr_z = curr_z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            traj.append(curr_z.clone())

        return torch.stack(traj, dim=0)

def compute_cfm_loss(flow_field: LatentFlowField, z_1: torch.Tensor,
                     z_0: Optional[torch.Tensor] = None,
                     t_0: float = 0.0, t_1: float = 1.0) -> torch.Tensor:
    """
    Computes Conditional Flow Matching loss given target endpoints.

    t_0 / t_1 place the segment on a global time axis, so several observed
    timepoints can train one shared field: a POD0->POD2 pair supervises the
    field near t=0 while a POD7->POD30 pair supervises it near t=1. The
    velocity target is divided by the segment width so it stays in units of
    latent distance per unit global time, which is what solve_rk4 integrates.
    Defaults reproduce the single-segment 0->1 behaviour exactly.
    """
    batch_size = z_1.size(0)
    device = z_1.device
    span = t_1 - t_0
    if not math.isfinite(span) or span <= 0:
        raise ValueError(f"t_1 must exceed t_0, got t_0={t_0}, t_1={t_1}")
    if z_1.ndim != 2 or batch_size == 0 or (z_0 is not None and z_0.shape != z_1.shape):
        raise ValueError('Endpoints must be nonempty matching batch-by-coordinate tensors')

    if z_0 is None:
        z_0 = torch.randn_like(z_1)

    u = torch.rand(batch_size, device=device)
    u_expanded = u.view(batch_size, 1)

    # Linear interpolation within the segment
    z_t = (1.0 - u_expanded) * z_0 + u_expanded * z_1
    target_velocity = (z_1 - z_0) / span

    pred_velocity = flow_field(z_t, t_0 + u * span)
    loss = F.mse_loss(pred_velocity, target_velocity)
    return loss

if __name__ == "__main__":
    print("=" * 60)
    print("Self-Testing Continuous Flow Matching & Neural ODE Engine...")
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    latent_dim = 8
    flow_net = LatentFlowField(latent_dim=latent_dim, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(flow_net.parameters(), lr=1e-3)
    
    # Target endpoint: clusters in latent space
    target_z1 = torch.randn(128, latent_dim, device=device) + 2.0
    
    # Train flow matching for 20 steps
    flow_net.train()
    for step in range(20):
        optimizer.zero_grad()
        loss = compute_cfm_loss(flow_net, target_z1)
        loss.backward()
        optimizer.step()
        
    print(f"CFM loss after 20 steps: {loss.item():.4f}")
    
    # Test numerical ODE trajectory solver
    flow_net.eval()
    integrator = FlowMatchingIntegrator(flow_net)
    z_start = torch.randn(10, latent_dim, device=device)
    trajectory = integrator.solve_rk4(z_start, n_steps=30)
    
    print(f"Trajectory shape [steps+1, batch, dim]: {trajectory.shape}")
    assert trajectory.shape == (31, 10, latent_dim), "ODE integration shape mismatch!"
    # The conditional field must actually use c: two contexts have to produce
    # different velocities at the same (z, t), or it cannot separate
    # individuals whose trajectories disagree.
    cfield = ConditionalLatentFlowField(latent_dim=8, context_dim=4).to(device)
    copt = torch.optim.Adam(cfield.parameters(), lr=1e-2)
    z0c = torch.randn(64, 8, device=device)
    ca = torch.zeros(4, device=device)
    cb = torch.ones(4, device=device)
    # Same start, opposite targets, distinguished only by the context vector.
    for _ in range(300):
        copt.zero_grad()
        la = compute_conditional_cfm_loss(cfield, z0c + 3.0, z0c, ca)
        lb = compute_conditional_cfm_loss(cfield, z0c - 3.0, z0c, cb)
        (la + lb).backward()
        copt.step()
    with torch.no_grad():
        va = cfield(z0c, 0.5, ca).mean().item()
        vb = cfield(z0c, 0.5, cb).mean().item()
    print(f"Conditional field: mean velocity ctx A = {va:+.3f}, ctx B = {vb:+.3f}")
    assert va > 1.0 > -1.0 > vb, (
        f"conditional field ignored its context: {va:+.3f} vs {vb:+.3f}")
    print("Conditional field separates opposite trajectories by context.")
    print("Flow Matching & Neural ODE Engine test PASSED!")
    print("=" * 60)
