"""Shared helpers for the finite-difference gate: grouping and relative error."""

from __future__ import annotations

import numpy as np


def relerr(a, b) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / (nb + 1e-30))


def group_vectors(grad: dict):
    """Return per-group flat vectors: combined W, tau_m, theta (and the W parts)."""
    w_all = np.concatenate([np.ravel(grad["W_in"]), np.ravel(grad["W2"]), np.ravel(grad["W_out"])])
    return {
        "W": w_all,
        "W_in": np.ravel(grad["W_in"]),
        "W2": np.ravel(grad["W2"]),
        "W_out": np.ravel(grad["W_out"]),
        "tau_m": np.array([grad["tau_m"]]),
        "theta": np.array([grad["theta"]]),
    }
