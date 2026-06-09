"""Spike-resolution velocity field, PF-ODE sampler, and 1-D Wasserstein-2.

The generation-time spiking resolution is a grid of T time-slots over the neuron
window: hidden spike times live on a 1/T grid (firing-rate quantization). Snapping
the readout-feeding spike times to that grid realises the deterministic O(1/T)
coding error of the lemma; as T grows the velocity field converges to the exact
(continuous-time) field, and the generated distribution tightens toward the target
along the a + b/T trend predicted for the spiking sampler.
"""

from __future__ import annotations

import numpy as np

from .forward import simulate
from .params import NetworkParams


def velocity(p: NetworkParams, u: np.ndarray, T: int | None = None,
             bisect_iters: int = 25) -> np.ndarray:
    """v_theta(x,t) = V_out(S). T=None: exact continuous time. T: spike times on a 1/T grid.

    bisect_iters=25 (looser than the gate's 60) is plenty here: the exact spike-time
    precision is far below the 1/T snap grid, so it only costs simulation time.
    """
    fwd = simulate(p, u, bisect_iters=bisect_iters)
    if T is None:
        return fwd.Vout_S
    dt_grid = p.S / T
    vout = np.zeros(p.m)
    for ev in fwd.events:
        if ev.layer == 2:
            s_snap = min(round(ev.time / dt_grid) * dt_grid, p.S)
            vout += p.W_out[:, ev.n] * np.exp(-(p.S - s_snap) / p.tau_out)
    return vout


def sample_pfode(p: NetworkParams, x0: np.ndarray, T: int | None = None,
                 n_steps: int = 40, bisect_iters: int = 25) -> np.ndarray:
    """Integrate dpsi/dt = v_theta(psi,t), psi(0)=x0, t:0->1 (Heun); return psi(1)."""
    psi = x0.astype(float).copy()
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = k * dt
        v1 = velocity(p, np.concatenate([psi, [t]]), T, bisect_iters)
        v2 = velocity(p, np.concatenate([psi + dt * v1, [min(t + dt, 1.0)]]), T, bisect_iters)
        psi = psi + 0.5 * dt * (v1 + v2)
    return psi


def w2_1d(samples_p: np.ndarray, samples_q: np.ndarray) -> float:
    """Exact Wasserstein-2 between two 1-D empirical distributions (equal counts)."""
    sp = np.sort(np.asarray(samples_p).ravel())
    sq = np.sort(np.asarray(samples_q).ravel())
    n = min(len(sp), len(sq))
    sp = sp[:: max(1, len(sp) // n)][:n]
    sq = sq[:: max(1, len(sq) // n)][:n]
    return float(np.sqrt(np.mean((sp - sq) ** 2)))


def velocity_quant_error(p: NetworkParams, test_inputs: np.ndarray, T: int) -> float:
    """Mean ||v^T(x,t) - v^inf(x,t)|| over test points (the coding error eps_quant(T))."""
    errs = []
    for u in test_inputs:
        errs.append(np.linalg.norm(velocity(p, u, T) - velocity(p, u, None)))
    return float(np.mean(errs))


def w2_to_gaussian(samples: np.ndarray, mu: float, sigma: float) -> float:
    """Exact 1-D Wasserstein-2 between empirical samples and the analytic N(mu, sigma^2).

    Uses the closed-form target quantile (mu + sigma * Phi^{-1}(u)) instead of a
    finite reference sample, so the target side contributes zero sampling noise.
    This isolates the spiking quantization (the O(1/T) term) from estimator noise:
    the only quantity varying with T is the empirical sample, not the target.
    """
    from scipy.stats import norm  # lazy: keeps the core/gate path pure-NumPy

    x = np.sort(np.asarray(samples, dtype=float).ravel())
    n = len(x)
    u = (np.arange(n) + 0.5) / n
    q = mu + sigma * norm.ppf(u)
    return float(np.sqrt(np.mean((x - q) ** 2)))
