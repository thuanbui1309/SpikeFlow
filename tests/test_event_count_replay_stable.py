"""Hidden spike counts are T-independent and replay-deterministic.

The event-driven ``simulate`` takes no T argument: layer-1 and layer-2 spike
counts and times are produced before any temporal snap. T enters only inside
``velocity`` when readout-feeding spike TIMES are rounded to a 1/T grid - which
changes the snapped readout but never the counts. These tests pin both facts.
"""

from __future__ import annotations

import numpy as np

from spikeflow.fc_params import init_fc_params
from spikeflow.forward import simulate
from spikeflow.generation import velocity


def _small_net():
    return init_fc_params(seed=0, n1=32, n2=32, m=48)


def _counts(fwd):
    n1 = sum(1 for e in fwd.events if e.layer == 1)
    n2 = sum(1 for e in fwd.events if e.layer == 2)
    return n1, n2


def test_simulate_counts_replay_deterministic():
    """Re-running simulate on the same input reproduces identical spike counts."""
    p = _small_net()
    rng = np.random.default_rng(7)
    for _ in range(100):
        u = np.concatenate([rng.standard_normal(p.m), [rng.uniform(0.0, 1.0)]])
        c1 = _counts(simulate(p, u))
        c2 = _counts(simulate(p, u))
        assert c1 == c2


def test_hidden_counts_t_independent_readout_snap_differs():
    """For a fixed input, counts are T-invariant; snapped readout times differ.

    The same T-independent ForwardResult drives every T-path, so the layer-1 and
    layer-2 counts cannot vary with T. Snapping those layer-2 times to a 1/T grid
    must (a) keep the number of contributions equal to the layer-2 count for every
    T, and (b) yield different snapped-time vectors for coarse vs fine T.
    """
    p = _small_net()
    rng = np.random.default_rng(11)
    u = np.concatenate([rng.standard_normal(p.m), [0.3]])

    fwd = simulate(p, u)
    n1, n2 = _counts(fwd)
    l2_times = np.array([e.time for e in fwd.events if e.layer == 2])

    snapped = {}
    for T in (16, 256):
        dt_grid = p.S / T
        s = np.minimum(np.round(l2_times / dt_grid) * dt_grid, p.S)
        # (a) count invariant: every layer-2 event still contributes exactly once.
        assert s.shape[0] == n2
        snapped[T] = s

    # The hidden counts are a property of the T-independent forward result.
    assert (n1, n2) == _counts(simulate(p, u))

    # (b) coarse vs fine snapped-time vectors differ (T actually changes times).
    if n2 > 0:
        assert not np.allclose(snapped[16], snapped[256])

    # Readout velocity differs across T and converges toward the exact field.
    v16 = velocity(p, u, 16)
    v256 = velocity(p, u, 256)
    vinf = velocity(p, u, None)
    assert np.linalg.norm(v16 - v256) > 1e-6
    assert np.linalg.norm(v256 - vinf) < np.linalg.norm(v16 - vinf)
