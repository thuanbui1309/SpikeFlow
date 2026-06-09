"""Central-difference gradient of the simulation-free flow-matching loss.

This is the empirical oracle for the gate. Because inter-spike dynamics are
propagated in closed form and crossings are bisected to machine precision, the
loss is an exact (integration-error-free) smooth function of the parameters away
from spike create/destroy boundaries, so central differences recover the true
continuous-time gradient up to truncation/round-off. Compared group-by-group
against the analytic adjoint.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .forward import simulate
from .params import NetworkParams


def _loss_at(p: NetworkParams, u: np.ndarray, target: np.ndarray, grid_dt: float) -> float:
    return simulate(p, u, grid_dt).loss(target)


def _array_grad(p, u, target, h, grid_dt, name):
    base = getattr(p, name)
    g = np.zeros_like(base)
    it = np.ndindex(base.shape)
    for idx in it:
        pp = p.copy()
        arr = getattr(pp, name)
        arr[idx] += h
        lp = _loss_at(pp, u, target, grid_dt)
        arr[idx] -= 2.0 * h
        lm = _loss_at(pp, u, target, grid_dt)
        g[idx] = (lp - lm) / (2.0 * h)
    return g


def _scalar_grad(p, u, target, h, grid_dt, name):
    val = getattr(p, name)
    lp = _loss_at(replace(p.copy(), **{name: val + h}), u, target, grid_dt)
    lm = _loss_at(replace(p.copy(), **{name: val - h}), u, target, grid_dt)
    return (lp - lm) / (2.0 * h)


def fd_gradient(p: NetworkParams, u: np.ndarray, target: np.ndarray,
                h: float = 1e-5, grid_dt: float = 0.05) -> dict:
    """Central-difference gradient over {W_in, W2, W_out, tau_m, theta}."""
    return {
        "W_in": _array_grad(p, u, target, h, grid_dt, "W_in"),
        "W2": _array_grad(p, u, target, h, grid_dt, "W2"),
        "W_out": _array_grad(p, u, target, h, grid_dt, "W_out"),
        "tau_m": _scalar_grad(p, u, target, h, grid_dt, "tau_m"),
        "theta": _scalar_grad(p, u, target, h, grid_dt, "theta"),
    }
