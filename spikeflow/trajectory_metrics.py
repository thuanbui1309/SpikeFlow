"""Stage-1 cadence-drift metric: do hidden spike COUNTS diverge between the
exact (continuous-time) PF-ODE trajectory and the snapped (1/T grid) trajectory?

The readout snap (generation.velocity with T=int) rounds layer-2 spike TIMES to a
1/T grid; at a FIXED input u the hidden spike counts are identical for every T
(simulate ignores T). So a PAIRED-same-state count comparison is 0 by construction
-- we log it as a sanity control that the harness reads counts correctly, NOT as a
signal.

The load-bearing signal is FREE-RUN cadence drift: rolling two INDEPENDENT Heun
trajectories from a shared x0 -- one exact (T=None), one snapped (T=int) -- the
snapped readout yields a different velocity, so the two state paths psi_k diverge
after step 0. A different psi feeds a different simulate(), which can produce a
different hidden count. A "flip" at step k = the integer total hidden count
(layer-1 + layer-2) at the snapped state differs from the count at the exact state.
Compounding of these flips over K steps is the metric: bounded/linear growth means
cadence is stable; super-linear growth means drift feeds drift.

The Heun update reproduces generation.sample_pfode byte-for-byte on the math
(dt=1/K, t=k*dt, min(t+dt,1) clamp, 0.5*dt*(v1+v2)); we re-implement it (rather than
call sample_pfode) only because sample_pfode returns psi(1) and never exposes psi_k
or per-step counts. Every velocity eval goes through generation.velocity so the
cross-step cache_guard tripwire stays integrated, and assert_no_survivors() runs at
each step boundary exactly as sample_pfode does.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from . import cache_guard
from .forward import simulate
from .generation import velocity
from .params import NetworkParams


def hidden_counts(p: NetworkParams, u: np.ndarray, T: int | None = None) -> tuple[int, int]:
    """(n_layer1, n_layer2) event counts at fixed input u (T-independent by design).

    Counts both hidden layers via the canonical train.py comprehension. T is
    accepted for call-site symmetry but does not affect simulate -- that
    T-invariance is exactly the property the paired control asserts.
    """
    return _counts_from_fwd(simulate(p, u))


def _counts_from_fwd(fwd) -> tuple[int, int]:
    n1 = sum(1 for e in fwd.events if e.layer == 1)
    n2 = sum(1 for e in fwd.events if e.layer == 2)
    return n1, n2


def _readout(p: NetworkParams, fwd, T: int | None) -> np.ndarray:
    """v_theta=V_out(S) from an existing fwd; mirrors generation.velocity:33-41 verbatim."""
    if T is None:
        return fwd.Vout_S
    dt_grid = p.S / T
    vout = np.zeros(p.m)
    for ev in fwd.events:
        if ev.layer == 2:
            s_snap = min(round(ev.time / dt_grid) * dt_grid, p.S)
            vout += p.W_out[:, ev.n] * np.exp(-(p.S - s_snap) / p.tau_out)
    return vout


def _vel_and_count(p: NetworkParams, u: np.ndarray, T: int | None,
                   bisect_iters: int) -> tuple[np.ndarray, int]:
    """One simulate -> (velocity at T, total hidden count); registers with cache_guard.

    Sharing one fwd between the rollout velocity and the count read avoids
    double-simulating the m=3072 net at each step (the spec's recommended path).
    """
    fwd = simulate(p, u, bisect_iters=bisect_iters)
    cache_guard.register(fwd)
    n1, n2 = _counts_from_fwd(fwd)
    return _readout(p, fwd, T), n1 + n2


def a2_violation_over_trajectory(p: NetworkParams, x0: np.ndarray, T: int,
                                 K: int, bisect_iters: int = 25) -> dict:
    """Roll an exact (T=None) and a snapped (T) Heun trajectory from the same x0.

    K Heun steps integrate t:0->1 (dt=1/K). At each step k (before the update,
    after assert_no_survivors) record:
      * paired flip  = count at exact-state under T-snap vs no-snap (== 0 control);
      * free-run flip = n_hidden(snapped_state) != n_hidden(exact_state) (the signal);
      * free-run count-diff = |n_hidden(snap) - n_hidden(exact)| (integer magnitude).
    Returns per-step arrays [K] plus the cumulative free-run flip curve. The v1
    evaluation point doubles as the count-read point so no extra simulate is spent.
    """
    psi_exact = x0.astype(float).copy()
    psi_snap = x0.astype(float).copy()
    dt = 1.0 / K
    paired = np.zeros(K, dtype=bool)
    freerun = np.zeros(K, dtype=bool)
    count_diff = np.zeros(K, dtype=int)
    for k in range(K):
        cache_guard.assert_no_survivors()
        t = k * dt
        u_exact = np.concatenate([psi_exact, [t]])
        u_snap = np.concatenate([psi_snap, [t]])
        # v1 + count from one shared simulate per state point.
        ve1, n_exact = _vel_and_count(p, u_exact, None, bisect_iters)
        vs1, n_snap = _vel_and_count(p, u_snap, T, bisect_iters)
        # Control: counts are T-independent at a fixed input -> verify once per
        # trajectory with a fresh paired simulate (T vs None) at the exact state.
        # Per-step would re-pay an m=3072 simulate for a property that holds globally.
        if k == 0:
            paired[k] = hidden_counts(p, u_exact, None) != hidden_counts(p, u_exact, T)
        # Signal: free-run divergence of counts between the two state paths.
        freerun[k] = n_snap != n_exact
        count_diff[k] = abs(n_snap - n_exact)
        # Heun corrector step, identical math to generation.sample_pfode.
        ve2 = velocity(p, np.concatenate([psi_exact + dt * ve1, [min(t + dt, 1.0)]]),
                       None, bisect_iters)
        psi_exact = psi_exact + 0.5 * dt * (ve1 + ve2)
        vs2 = velocity(p, np.concatenate([psi_snap + dt * vs1, [min(t + dt, 1.0)]]),
                       T, bisect_iters)
        psi_snap = psi_snap + 0.5 * dt * (vs1 + vs2)
    cache_guard.assert_no_survivors()
    return {
        "paired_flip": paired,
        "freerun_flip": freerun,
        "count_diff": count_diff,
        "cumulative_flip": np.cumsum(freerun.astype(int)),
        "violation_rate": float(freerun.mean()),
        "paired_max": int(paired.sum()),
    }


_WORKER_P: NetworkParams | None = None


def _init_worker(p: NetworkParams) -> None:
    """Pool initializer: pin the network in each worker once (not per task)."""
    global _WORKER_P
    _WORKER_P = p


def _traj_worker(args):
    x0, T, K, bisect_iters = args
    return a2_violation_over_trajectory(_WORKER_P, x0, T, K, bisect_iters)


def aggregate_violation(p: NetworkParams, T: int, K: int, n_samples: int,
                        seed: int = 0, n_workers: int = 1,
                        bisect_iters: int = 25) -> dict:
    """Mean free-run violation_rate + cumulative-flip curve over n_samples x0 seeds.

    Each sample is one full paired trajectory pair; the pool parallel axis is the
    sample. cache_guard is process-local module state, safe under mp.Pool (each
    worker has its own _live_results and runs a self-contained rollout).
    """
    x0_batch = np.random.default_rng(seed).standard_normal((n_samples, p.m))
    if n_workers <= 1:
        outs = [a2_violation_over_trajectory(p, x0_batch[i], T, K, bisect_iters)
                for i in range(n_samples)]
    else:
        args = [(x0_batch[i], T, K, bisect_iters) for i in range(n_samples)]
        chunk = max(1, len(args) // (n_workers * 4))
        with mp.Pool(n_workers, initializer=_init_worker, initargs=(p,)) as pool:
            outs = pool.map(_traj_worker, args, chunksize=chunk)

    freerun = np.stack([o["freerun_flip"] for o in outs])      # [n, K] bool
    cum = np.stack([o["cumulative_flip"] for o in outs])       # [n, K] int
    paired_max = max(o["paired_max"] for o in outs)
    return {
        "violation_rate": float(freerun.mean()),               # over (sample, step)
        "cumulative_flip_curve": cum.mean(axis=0),             # [K] mean cumulative flips
        "per_sample_rate": freerun.mean(axis=1),               # [n]
        "paired_control_max": paired_max,                      # MUST be 0
        "n_samples": n_samples,
        "T": T,
        "K": K,
    }
