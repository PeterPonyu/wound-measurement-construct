"""Conditional flow-matching family trained only on permitted cells.

Segments join consecutive permitted times within each training individual (for a
global missing time the segment bridges the held time).  Couplings: independent
endpoints (the existing studies), exact minibatch optimal transport, and an
entropic Schrodinger-bridge coupling with Brownian-bridge noise.  Prediction
integrates the learned field with fixed-step RK4 on the shared clock.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

from ...common.rng import derive_seed
from ..infosets import Info
from .base import Method, MethodSpec, register

MISSING_TIME = Info.OWN_SOURCE | Info.OWN_OTHER_TIMES | Info.OTHERS_NONTARGET_TIMES
LEAVE_INDIVIDUAL = Info.OWN_SOURCE | Info.OTHERS_NONTARGET_TIMES | Info.OTHERS_TARGET_TIME
VARIANTS = (("missing_time", MISSING_TIME), ("leave_individual", LEAVE_INDIVIDUAL))
DEFAULTS = dict(hidden=256, depth=3, time_dim=32, steps=10_000, batch_size=512, lr=1e-3,
                rk4_steps_per_interval=50, grad_clip=1.0, sb_sigma_scale=0.25, sinkhorn_iterations=300,
                sinkhorn_tolerance=1e-3, context_dropout=0.2)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        half = max(dim // 2, 1)
        self.register_buffer("frequencies", torch.exp(torch.linspace(0.0, math.log(1000.0), half)))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        angles = s.reshape(-1, 1) * self.frequencies.reshape(1, -1)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class VelocityField(nn.Module):
    def __init__(self, dim: int, *, hidden: int = 256, depth: int = 3, time_dim: int = 32, context_dim: int = 0):
        super().__init__()
        self.embed = TimeEmbedding(time_dim)
        self.context_dim = context_dim
        width = dim + 2 * max(time_dim // 2, 1) + context_dim
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        layers.append(nn.Linear(width, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z, s, context=None):
        parts = [z, self.embed(s)]
        if self.context_dim:
            parts.append(context)
        return self.net(torch.cat(parts, dim=1))


def training_segments(train: dict):
    by_individual = {}
    for (individual, time), cells in train.items():
        by_individual.setdefault(individual, {})[float(time)] = cells
    segments = []
    for individual in sorted(by_individual):
        times = sorted(by_individual[individual])
        segments.extend((individual, a, b) for a, b in zip(times[:-1], times[1:]))
    if not segments:
        raise ValueError("no training segments: each individual needs at least two permitted times")
    return segments, by_individual


def _sample_rows(x: torch.Tensor, n: int, generator: torch.Generator) -> torch.Tensor:
    count = x.shape[0]
    index = torch.randperm(count, generator=generator)[:n] if count >= n else torch.randint(count, (n,), generator=generator)
    return x[index.to(x.device)]


def _sinkhorn_log_plan(cost: torch.Tensor, epsilon: float, iterations: int, tolerance: float,
                       stage_iterations: int = 50, *, return_diagnostics: bool = False):
    """Log-domain Sinkhorn with uniform marginals and epsilon scaling (warm-started from a large epsilon).

    The legacy row-only stopping rule is deliberately unchanged. Optional
    diagnostics measure both marginals of the *returned* plan and distinguish
    warm-start work from target-epsilon work. Exhaustion is reported, not hidden.
    """
    if (not isinstance(cost, torch.Tensor) or cost.ndim != 2 or not all(cost.shape)
            or not cost.is_floating_point() or not bool(torch.isfinite(cost).all())
            or bool((cost < 0).any())):
        raise ValueError("Sinkhorn cost must be a nonempty finite nonnegative floating matrix")
    for name, value in (("epsilon", epsilon), ("tolerance", tolerance)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Sinkhorn {name} must be finite and positive")
    for name, value in (("iterations", iterations), ("stage_iterations", stage_iterations)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Sinkhorn {name} must be a positive integer")
    n, m = cost.shape
    log_a = torch.full((n,), -math.log(n), device=cost.device, dtype=cost.dtype)
    log_b = torch.full((m,), -math.log(m), device=cost.device, dtype=cost.dtype)
    f = torch.zeros(n, device=cost.device, dtype=cost.dtype)
    g = torch.zeros(m, device=cost.device, dtype=cost.dtype)
    schedule, current = [], max(float(cost.mean()), float(epsilon))
    if not math.isfinite(current):
        raise ValueError('Sinkhorn mean cost overflowed the working dtype')
    while current > epsilon:
        schedule.append(current)
        current /= 2.0
    schedule.append(float(epsilon))
    error, stages = float("inf"), []
    for stage, eps in enumerate(schedule):
        final = stage == len(schedule) - 1
        budget = int(iterations) if final else int(stage_iterations)
        for iteration in range(budget):
            f = eps * (log_a - torch.logsumexp((g[None, :] - cost) / eps, dim=1))
            g = eps * (log_b - torch.logsumexp((f[:, None] - cost) / eps, dim=0))
            if iteration % 10 == 9 or iteration == budget - 1:
                rows = torch.logsumexp((f[:, None] + g[None, :] - cost) / eps, dim=1)
                error = float((rows.exp() - log_a.exp()).abs().sum())
                if not math.isfinite(error):
                    raise FloatingPointError("nonfinite Sinkhorn residual")
                if error < tolerance:
                    break
        if return_diagnostics:
            stages.append({"epsilon": eps, "target_stage": final, "budget": budget,
                           "iterations": iteration + 1, "legacy_row_l1": error,
                           "legacy_stop_satisfied": error < tolerance,
                           "budget_exhausted": iteration + 1 == budget})
    log_plan = (f[:, None] + g[None, :] - cost) / float(epsilon)
    if not bool(torch.isfinite(log_plan).all()):
        raise FloatingPointError("nonfinite Sinkhorn returned log plan")
    if not return_diagnostics:
        return log_plan, error
    probability = log_plan.exp()
    row_l1 = float((probability.sum(1) - log_a.exp()).abs().sum())
    column_l1 = float((probability.sum(0) - log_b.exp()).abs().sum())
    mass = float(probability.sum())
    if not all(math.isfinite(v) for v in (row_l1, column_l1, mass)):
        raise FloatingPointError("nonfinite Sinkhorn returned plan diagnostics")
    converged = max(row_l1, column_l1) <= tolerance
    diagnostics = {"row_l1": row_l1, "column_l1": column_l1, "mass": mass,
                   "finite_log_plan": True, "finite_plan": True, "tolerance": tolerance,
                   "epsilon": float(epsilon), "shape": [n, m], "dtype": str(cost.dtype),
                   "legacy_reported_row_l1": error, "stages": stages,
                   "target_iterations": stages[-1]["iterations"],
                   "warm_iterations": sum(s["iterations"] for s in stages[:-1]),
                   "total_iterations": sum(s["iterations"] for s in stages),
                   "target_budget_exhausted": stages[-1]["budget_exhausted"],
                   "both_marginals_converged": converged,
                   "nonconverged_at_target_budget": not converged and stages[-1]["budget_exhausted"],
                   "stopping_rule": "legacy_row_l1_strict_less_than_tolerance_every_10_or_last"}
    return log_plan, error, diagnostics


def _couple(x0, x1, kind: str, generator: torch.Generator, p: dict, epsilon: float | None = None,
            *, return_diagnostics: bool = False):
    """Return (re-indexed x1, row-marginal error of the entropic plan or None)."""
    if kind == "independent":
        return x1, None
    if kind == "ot":
        import ot

        a = np.full(len(x0), 1.0 / len(x0))
        b = np.full(len(x1), 1.0 / len(x1))
        cost = ot.dist(x0.detach().cpu().double().numpy(), x1.detach().cpu().double().numpy())
        index = torch.as_tensor(ot.emd(a, b, cost).argmax(axis=1), device=x1.device)
        return x1[index], None
    if kind == "sb":
        # Static Schrodinger-bridge coupling: entropic OT with the segment's epsilon = 2 sigma^2 on squared
        # Euclidean cost; pairs are drawn exactly from the row-conditional plan by the Gumbel-max trick.
        cost = torch.cdist(x0.detach(), x1.detach()) ** 2
        result = _sinkhorn_log_plan(cost, float(epsilon), p["sinkhorn_iterations"], p["sinkhorn_tolerance"],
                                    return_diagnostics=return_diagnostics)
        log_plan, error = result[:2]
        uniform = torch.rand(log_plan.shape, generator=generator).to(log_plan.device).clamp_min(1e-12)
        index = torch.argmax(log_plan - torch.log(-torch.log(uniform)), dim=1)
        return x1[index], result[2] if return_diagnostics else error
    raise ValueError(f"unknown coupling {kind!r}")


class FlowMatching(Method):
    coupling = "independent"
    conditioned = False

    def _fit(self, inputs, *, seed: int, device: str) -> dict:
        begun, timings = time.perf_counter(), {'coupling': 0., 'optimizer': 0.}
        p = {**DEFAULTS, **self.params}
        segments, by_individual = training_segments(inputs.train)
        self.device = torch.device(device)
        torch.manual_seed(derive_seed("flow-init", seed))
        generator = torch.Generator().manual_seed(derive_seed("flow-batches", seed))
        tensors = {(ind, t): torch.as_tensor(np.asarray(cells, dtype=np.float32), device=self.device)
                   for ind, times in by_individual.items() for t, cells in times.items()}
        dim = next(iter(tensors.values())).shape[1]
        contexts = ({key: torch.cat([x.mean(0), x.std(0, unbiased=False)]) for key, x in tensors.items()}
                    if self.conditioned else {})
        self.field = VelocityField(dim, hidden=p["hidden"], depth=p["depth"], time_dim=p["time_dim"],
                                   context_dim=2 * dim if self.conditioned else 0).to(self.device)
        optimizer = torch.optim.Adam(self.field.parameters(), lr=p["lr"])
        clock, batch, losses, worst_marginal = inputs.clock, int(p["batch_size"]), [], 0.0
        order, cursor = torch.randperm(len(segments), generator=generator).tolist(), 0
        bridge, sb_batches = {}, []
        if self.coupling == "sb":
            # Bridge noise per coordinate is a fixed fraction of the segment's within-group standard deviation
            # (pooled over source and target pilot batches), so it does not grow with the dimension; the static
            # Schrodinger-bridge coupling then uses entropic regularization epsilon = 2 sigma^2.
            for segment in segments:
                individual, a, b = segment
                pilot0 = _sample_rows(tensors[(individual, a)], min(batch, 512), generator)
                pilot1 = _sample_rows(tensors[(individual, b)], min(batch, 512), generator)
                spread = math.sqrt(0.5 * float(pilot0.var(0, unbiased=False).mean() + pilot1.var(0, unbiased=False).mean()))
                sigma = float(p["sb_sigma_scale"]) * max(spread, 1e-6)
                bridge[segment] = (2.0 * sigma ** 2, sigma)
        profile = p.get('profile_stages', False)

        def tick():
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            return time.perf_counter()
        if profile:
            timings['setup_including_bridge_pilots'] = tick() - begun
        for step in range(int(p["steps"])):
            if cursor == len(order):
                order, cursor = torch.randperm(len(segments), generator=generator).tolist(), 0
            segment = segments[order[cursor]]
            individual, a, b = segment
            cursor += 1
            x0 = _sample_rows(tensors[(individual, a)], batch, generator)
            epsilon, sigma = bridge.get(segment, (None, 0.0))
            started = tick() if profile else None
            x1, marginal_error = _couple(x0, _sample_rows(tensors[(individual, b)], batch, generator), self.coupling,
                                         generator, p, epsilon, return_diagnostics=True)
            if profile:
                now = tick()
                timings['coupling'] += now - started
                started = now
            if marginal_error is not None:
                worst_marginal = max(worst_marginal, marginal_error["legacy_reported_row_l1"])
                sb_batches.append({"step": step + 1, "individual": individual, "source_time": a,
                                   "target_time": b, **marginal_error})
            lam = torch.rand(batch, 1, generator=generator).to(self.device)
            span = clock[b] - clock[a]
            centre = (1 - lam) * x0 + lam * x1
            if self.coupling == "sb":
                noise = torch.randn(x0.shape, generator=generator).to(self.device)
                z = centre + sigma * torch.sqrt(lam * (1 - lam)) * noise
                target = (1 - 2 * lam) / (2 * lam * (1 - lam) + 1e-8) * (z - centre) + (x1 - x0)
            else:
                z, target = centre, x1 - x0
            s = (clock[a] + lam.reshape(-1) * span).float()
            context = None
            if self.conditioned:
                keep = (torch.rand(batch, 1, generator=generator) >= p["context_dropout"]).float().to(self.device)
                context = contexts[(individual, a)].unsqueeze(0).expand(batch, -1) * keep
            loss = torch.mean((self.field(z, s, context) - target / span) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite flow-matching loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if p["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(self.field.parameters(), p["grad_clip"])
            optimizer.step()
            if profile:
                timings['optimizer'] += tick() - started
            if step % 100 == 0 or step == int(p["steps"]) - 1:
                losses.append(float(loss.detach().cpu()))
        self.field.eval()
        spacing = np.diff(sorted(clock.values()))
        self.interval = float(spacing.min()) if spacing.size else 1.0
        info = {"segments": len(segments), "steps": int(p["steps"]), "loss_first": losses[0], "loss_last": losses[-1],
                "loss_trace_every_100": losses, "parameters": int(sum(q.numel() for q in self.field.parameters()))}
        if profile:
            info['stage_seconds_synchronized'] = timings
            info['timing_scope'] = 'diagnostic_with_synchronization_overhead_not_equal_wall_budget'
        if self.coupling == "sb":
            epsilons = np.array([value[0] for value in bridge.values()])
            residuals = np.asarray([[d["row_l1"], d["column_l1"]] for d in sb_batches])
            info.update(sb_max_marginal_error=worst_marginal, sb_sigma_scale=float(p["sb_sigma_scale"]),
                        sb_epsilon_range=[float(epsilons.min()), float(epsilons.max())],
                        sb_sigma_range=[float(np.sqrt(epsilons.min() / 2)), float(np.sqrt(epsilons.max() / 2))],
                        sb_batch_diagnostics=sb_batches,
                        sb_solver_summary={"batches": len(sb_batches),
                            "converged_batches": sum(d["both_marginals_converged"] for d in sb_batches),
                            "target_budget_exhausted_batches": sum(d["target_budget_exhausted"] for d in sb_batches),
                            "nonconverged_batches": sum(not d["both_marginals_converged"] for d in sb_batches),
                            "total_iterations": sum(d["total_iterations"] for d in sb_batches),
                            "max_row_l1": float(residuals[:, 0].max()),
                            "max_column_l1": float(residuals[:, 1].max()),
                            "row_column_quantiles": {str(q): np.quantile(residuals, q, axis=0).tolist()
                                                     for q in (0, .5, .9, .95, 1)},
                            "scope": "coupling_marginals_only_not_optimizer_or_biological_convergence"})
        return info

    def _predict(self, source, *, source_time, target_time, **_):
        p = {**DEFAULTS, **self.params}
        s0, s1 = self.inputs.clock[source_time], self.inputs.clock[target_time]
        steps = max(int(math.ceil(p["rk4_steps_per_interval"] * abs(s1 - s0) / self.interval)), 1)
        dt = (s1 - s0) / steps
        x = torch.as_tensor(np.asarray(source, dtype=np.float32), device=self.device)
        context = torch.cat([x.mean(0), x.std(0, unbiased=False)]) if self.conditioned else None
        blocks = []
        with torch.no_grad():
            for start in range(0, len(x), 8192):
                z = x[start:start + 8192]
                c = context.unsqueeze(0).expand(len(z), -1) if self.conditioned else None

                def velocity(state, time):
                    return self.field(state, torch.full((len(state),), float(time), device=self.device), c)

                for k in range(steps):
                    s = s0 + k * dt
                    k1 = velocity(z, s)
                    k2 = velocity(z + 0.5 * dt * k1, s + 0.5 * dt)
                    k3 = velocity(z + 0.5 * dt * k2, s + 0.5 * dt)
                    k4 = velocity(z + dt * k3, s + dt)
                    z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                blocks.append(z.cpu().numpy())
        return np.vstack(blocks)


@register
class CFMIndependent(FlowMatching):
    spec = MethodSpec("cfm_independent", "flow", VARIANTS, True, True,
                      "Conditional flow matching with independent same-individual endpoints.")
    coupling = "independent"


@register
class CFMOT(FlowMatching):
    spec = MethodSpec("cfm_ot", "flow", VARIANTS, True, True,
                      "Conditional flow matching with exact minibatch optimal-transport coupling.")
    coupling = "ot"


@register
class CFMSB(FlowMatching):
    spec = MethodSpec("cfm_sb", "flow", VARIANTS, True, True,
                      "Schrodinger-bridge flow matching (entropic coupling, Brownian-bridge noise).")
    coupling = "sb"


@register
class CFMConditioned(FlowMatching):
    spec = MethodSpec("cfm_conditioned", "flow", VARIANTS, True, True,
                      "Independent-coupling flow conditioned on the source mean and standard deviation.")
    coupling = "independent"
    conditioned = True
