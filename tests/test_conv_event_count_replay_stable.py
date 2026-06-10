"""Conv velocity net keeps spike counts exact under weight sharing + pooling.

The convolution is materialised as a structured dense net and run on the same event
engine, so the dense-net invariants must survive: replay determinism (deterministic
tie-break under shared kernels), hidden-count T-invariance, and non-vacuity (both
hidden layers actually fire). The tie audit must run and report a finite gap.
"""

from __future__ import annotations

import numpy as np

from spikeflow.conv_event_identity import (
    audit_event_identity,
    event_counts,
    min_crossing_gap,
    replay_stable,
    tie_stress,
)
from spikeflow.conv_params import init_conv_params


def _tiny_conv():
    # (3,8,8) image, weight-shared k=4/stride4 -> (3,2,2), pool k2/stride2 -> (3,1,1).
    p, spec = init_conv_params(seed=0, img=(3, 8, 8), c1=3, k1=4, stride1=4,
                               c2=3, k2=2, stride2=2)
    return p, spec


def _u(rng, p):
    return np.concatenate([rng.standard_normal(p.m), [rng.uniform(0.0, 1.0)]])


def test_conv_counts_replay_deterministic():
    """Re-simulating a fixed input reproduces identical conv hidden counts."""
    p, _ = _tiny_conv()
    rng = np.random.default_rng(3)
    for _ in range(40):
        u = _u(rng, p)
        assert replay_stable(p, u, n_replays=4)


def test_conv_hidden_layers_non_vacuous():
    """Both conv hidden layers fire (a silent net would pass every gate vacuously)."""
    p, _ = _tiny_conv()
    rng = np.random.default_rng(5)
    n1, n2 = event_counts(p, _u(rng, p))
    assert n1 > 0 and n2 > 0


def test_conv_counts_t_invariant_and_tie_audit_runs():
    """Counts are T-agnostic and the tie audit returns a finite, non-degenerate gap."""
    p, _ = _tiny_conv()
    rng = np.random.default_rng(9)
    inputs = np.stack([_u(rng, p) for _ in range(12)])
    res = audit_event_identity(p, inputs, Ts=(16, 256), tie_eps=1e-9)
    assert res["replay_stable"] is True
    assert res["counts_t_invariant"] is True
    # Random continuous inputs do not produce machine-zero ties.
    assert res["global_min_gap"] > 1e-9
    assert res["near_tie_rate"] == 0.0
    assert res["mean_n1"] > 0 and res["mean_n2"] > 0


def test_min_crossing_gap_keys():
    """min_crossing_gap exposes per-layer gap + count for the report."""
    p, _ = _tiny_conv()
    rng = np.random.default_rng(13)
    g = min_crossing_gap(p, _u(rng, p))
    assert {"min_gap_l1", "min_gap_l2", "n_l1", "n_l2"} <= set(g)


def test_tie_stress_counts_stay_exact_in_tie_regime():
    """Uniform-image tie stress keeps counts deterministic + order-benign.

    Spatially uniform inputs push weight-shared neurons toward simultaneous crossings;
    the count must remain deterministic on replay and unchanged when a tiny asymmetric
    ramp breaks the ties (the synaptic update commutes over simultaneous spikes). The
    crossing gap stays finite -- per-neuron bias noise prevents exact (machine-zero)
    ties -- and far below the readout snap grid, so ties cannot reach the snapped output.
    """
    p, _ = _tiny_conv()
    ts = tie_stress(p, img=(3, 8, 8))
    assert ts["tie_replay_stable"] is True
    assert ts["perturb_count_change_rate"] == 0.0
    assert 0.0 < ts["tie_min_gap"] < float("inf")
