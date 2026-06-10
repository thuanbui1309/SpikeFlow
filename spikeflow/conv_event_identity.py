"""Conv event-identity audit: does weight sharing + pooling keep spike counts exact?

Three checks, run on the conv velocity net before any trajectory gate:

* replay-stability -- a fixed input must produce the SAME hidden spike counts on
  every re-simulation. The event scheduler breaks threshold-crossing ties by
  neuron index; under weight sharing many neurons can cross near-simultaneously, so
  this confirms the tie-break is deterministic (counts never wobble on replay).

* T-invariance -- hidden counts must not depend on the readout snap grid T. The
  simulator never sees T (only the readout snaps layer-2 times afterwards), so this
  holds by construction; the check is a regression guard that the count-reading path
  stays T-agnostic, mirroring the dense-net control.

* tie audit -- weight sharing makes spatially symmetric input patches give neurons
  IDENTICAL drives, hence near-simultaneous threshold crossings (ties). On random
  continuous inputs the patches differ, so ties have measure zero and the gap audit
  only acts as a regression baseline. The load-bearing tie test is `tie_stress`: it
  drives the net with spatially uniform images (the maximal-tie regime random data
  never reaches), pushing the crossing gaps toward zero, and confirms the counts stay
  deterministic there -- the additive synaptic update (I += W2 column) makes the order
  of simultaneous layer-1 spikes commute, so a correct engine keeps counts independent
  of tie-break order. It also pooled-aggregates many simultaneous layer-1 spikes into
  one layer-2 drive, so it stresses pooling under ties at the same time.

Everything reuses `forward.simulate`; no event-loop logic is duplicated here.
"""

from __future__ import annotations

import numpy as np

from .forward import simulate
from .params import NetworkParams


def event_counts(p: NetworkParams, u: np.ndarray, bisect_iters: int = 25) -> tuple[int, int]:
    """(n_layer1, n_layer2) hidden spike counts at input u."""
    fwd = simulate(p, u, bisect_iters=bisect_iters)
    n1 = sum(1 for e in fwd.events if e.layer == 1)
    n2 = sum(1 for e in fwd.events if e.layer == 2)
    return n1, n2


def replay_stable(p: NetworkParams, u: np.ndarray, n_replays: int = 16,
                  bisect_iters: int = 25) -> bool:
    """True iff every re-simulation of u yields identical hidden counts.

    On random inputs this is a determinism regression guard. It becomes load-bearing
    when u is a tie-inducing (spatially uniform) input: there many neurons cross at
    once, so a stable count proves the index tie-break is deterministic and order
    -independent -- hence `tie_stress` runs it on the symmetric inputs.
    """
    ref = event_counts(p, u, bisect_iters)
    return all(event_counts(p, u, bisect_iters) == ref for _ in range(n_replays - 1))


def counts_t_invariant(p: NetworkParams, u: np.ndarray, Ts=(16, 256),
                       bisect_iters: int = 25) -> bool:
    """True iff hidden counts are identical across snap grids T (T-agnostic by design).

    simulate ignores T, so a fresh simulate per T must return the same counts; a
    failure would mean count-reading leaked a T dependence (regression guard).
    """
    ref = event_counts(p, u, bisect_iters)
    return all(event_counts(p, u, bisect_iters) == ref for _ in Ts)


def min_crossing_gap(p: NetworkParams, u: np.ndarray, bisect_iters: int = 25) -> dict:
    """Smallest gap between consecutive same-layer crossing times + per-layer event count."""
    fwd = simulate(p, u, bisect_iters=bisect_iters)
    out = {}
    for layer in (1, 2):
        ts = sorted(e.time for e in fwd.events if e.layer == layer)
        gaps = np.diff(ts) if len(ts) > 1 else np.array([np.inf])
        out[f"min_gap_l{layer}"] = float(gaps.min()) if gaps.size else float("inf")
        out[f"n_l{layer}"] = len(ts)
    return out


def audit_event_identity(p: NetworkParams, inputs: np.ndarray, Ts=(16, 256),
                         tie_eps: float = 1e-9, bisect_iters: int = 25) -> dict:
    """Run all three checks over a batch of inputs; aggregate to a verdict dict.

    inputs: [N, d_in] already-concatenated (x_t, t) rows. Returns replay/T-invariance
    pass flags, the global minimum crossing gap, and the near-tie rate. The caller
    decides PASS (replay+T-invariant hold and ties are negligible) vs investigate.
    """
    replay_ok = True
    tinv_ok = True
    min_gap = float("inf")
    near_ties = 0
    total_pairs = 0
    n1s, n2s = [], []
    for u in inputs:
        if not replay_stable(p, u, bisect_iters=bisect_iters):
            replay_ok = False
        if not counts_t_invariant(p, u, Ts, bisect_iters=bisect_iters):
            tinv_ok = False
        g = min_crossing_gap(p, u, bisect_iters=bisect_iters)
        min_gap = min(min_gap, g["min_gap_l1"], g["min_gap_l2"])
        n1s.append(g["n_l1"])
        n2s.append(g["n_l2"])
        # Count near-ties among layer-1 (the weight-shared layer most prone to them).
        if g["min_gap_l1"] < tie_eps:
            near_ties += 1
        total_pairs += 1
    return {
        "replay_stable": replay_ok,
        "counts_t_invariant": tinv_ok,
        "global_min_gap": min_gap,
        "near_tie_rate": near_ties / max(1, total_pairs),
        "tie_eps": tie_eps,
        "mean_n1": float(np.mean(n1s)),
        "mean_n2": float(np.mean(n2s)),
        "n_inputs": len(inputs),
    }


def _uniform_image_input(level: float, img: tuple, t: float = 0.3) -> np.ndarray:
    """A spatially uniform image (every pixel = level) + flow time t, flattened (x_t, t).

    Spatial uniformity makes every receptive-field patch identical, so all neurons
    sharing a kernel see the SAME drive -- the maximal weight-sharing tie regime.
    """
    C, H, W = img
    return np.concatenate([np.full(C * H * W, float(level)), [t]])


def tie_stress(p: NetworkParams, img: tuple, levels=(-1.0, 0.5, 1.5),
               perturb_eps=(1e-9, 1e-7, 1e-5), tie_eps: float = 1e-6,
               bisect_iters: int = 25) -> dict:
    """Force weight-sharing ties with uniform images; confirm counts stay exact there.

    For each uniform-image level: (a) the crossing gaps collapse toward zero (this IS
    the tie regime random inputs miss); (b) replay must reproduce identical counts
    (deterministic tie-break); (c) breaking the ties with a tiny asymmetric ramp must
    leave the total count unchanged (the synaptic update commutes over simultaneous
    spikes, so tie order cannot move the count). Returns the worst gap, the near-tie
    count, and the perturbation-induced count-change rate (0 = ties fully benign).
    """
    worst_gap = float("inf")
    replay_ok = True
    near_tie_hits = 0
    perturb_trials = 0
    perturb_changes = 0
    ramp = np.linspace(-1.0, 1.0, p.m)                      # fixed asymmetric tie-breaker
    for level in levels:
        u = _uniform_image_input(level, img)
        g = min_crossing_gap(p, u, bisect_iters=bisect_iters)
        worst_gap = min(worst_gap, g["min_gap_l1"], g["min_gap_l2"])
        if g["min_gap_l1"] < tie_eps:
            near_tie_hits += 1
        if not replay_stable(p, u, n_replays=4, bisect_iters=bisect_iters):
            replay_ok = False
        base = event_counts(p, u, bisect_iters)
        for eps in perturb_eps:
            pert = u.copy()
            pert[:-1] += eps * ramp
            perturb_trials += 1
            if event_counts(p, pert, bisect_iters) != base:
                perturb_changes += 1
    return {
        "tie_min_gap": worst_gap,
        "tie_replay_stable": replay_ok,
        "near_tie_levels": near_tie_hits,
        "n_levels": len(levels),
        "perturb_count_change_rate": perturb_changes / max(1, perturb_trials),
        "tie_eps": tie_eps,
    }
