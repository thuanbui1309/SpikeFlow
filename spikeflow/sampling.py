"""Flow-matching sample generation (simulation-free, single velocity evaluation).

A training sample draws noise x0 ~ N(0,I), data x1 ~ q, and a transport time
t ~ U[0,1], forms the linear interpolant x_t = (1-t) x0 + t x1, and targets the
constant conditional velocity Delta = x1 - x0. The loss evaluates the velocity
field exactly once at (x_t, t); there is no integration of a generative ODE here,
which is what keeps training cheap and the adjoint a single backward sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FlowSample:
    u: np.ndarray       # network input (x_t, t), shape [d+1]
    target: np.ndarray  # Delta = x1 - x0, shape [d]
    x0: np.ndarray
    x1: np.ndarray
    t: float


def sample_flow_matching(seed: int, d: int = 2, data_mean: float = 2.0) -> FlowSample:
    """One (x_t, t) -> Delta training pair with a fixed RNG seed.

    Data q is a shifted Gaussian by default (a stand-in target for the gate; the
    toy Wasserstein experiment uses a richer q). The seed makes the sample fixed
    and reproducible so the finite-difference comparison is deterministic.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(d)
    x1 = data_mean + rng.standard_normal(d)
    t = float(rng.uniform(0.0, 1.0))
    x_t = (1.0 - t) * x0 + t * x1
    u = np.concatenate([x_t, [t]])
    return FlowSample(u=u, target=x1 - x0, x0=x0, x1=x1, t=t)
