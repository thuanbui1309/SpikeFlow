"""The PF-ODE sampler must re-simulate hidden dynamics fresh every step.

A correct ``sample_pfode`` lets each per-step ForwardResult die before the next
step (CPython refcounting), so the cross-step cache tripwire never fires. If a
caller holds a strong reference to a result across a step, the tripwire raises.
"""

from __future__ import annotations

import numpy as np
import pytest

from spikeflow import cache_guard
from spikeflow.fc_params import init_fc_params
from spikeflow.generation import sample_pfode, velocity


def _small_net():
    # m = 4*4*3 = 48: a tiny but identical-code-path version of the wide-FC net.
    return init_fc_params(seed=0, n1=32, n2=32, m=48)


def test_no_cross_step_cache():
    """Clean sampler over multiple steps: no ForwardResult survives a step."""
    p = _small_net()
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal(p.m)
    # Should complete without the tripwire raising.
    out = sample_pfode(p, x0, T=64, n_steps=10)
    assert out.shape == (p.m,)
    assert np.all(np.isfinite(out))
    # After a clean run the live set is empty at the final boundary too.
    cache_guard.assert_no_survivors()


def test_cross_step_cache_raises():
    """Holding a forward result alive across a step trips the guard."""
    p = _small_net()
    rng = np.random.default_rng(2)
    u = np.concatenate([rng.standard_normal(p.m), [0.3]])

    cache_guard.assert_no_survivors()  # start from a clean boundary

    held = []  # persistent container that keeps the result strongly referenced

    class _LeakyResult:
        """A ForwardResult-like object stashed so its weakref stays alive."""

    leaked = _LeakyResult()
    cache_guard.register(leaked)  # simulate a velocity() call's registration
    held.append(leaked)           # ... but a caller caches it across the step

    with pytest.raises(RuntimeError):
        cache_guard.assert_no_survivors()

    # Drop the strong reference; the boundary is clean again.
    held.clear()
    cache_guard.assert_no_survivors()


def test_velocity_does_not_leak_forward_result():
    """A single velocity() call leaves no surviving result at the next boundary."""
    p = _small_net()
    rng = np.random.default_rng(3)
    u = np.concatenate([rng.standard_normal(p.m), [0.1]])
    cache_guard.assert_no_survivors()
    _ = velocity(p, u, T=32)
    # The local fwd inside velocity went out of scope -> nothing survives.
    cache_guard.assert_no_survivors()
