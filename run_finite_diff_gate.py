"""Finite-difference gate for the exact spiking velocity-field gradient.

Certifies that the backward-adjoint gradient of the simulation-free flow-matching
loss matches central finite differences group-by-group (weights, tau_m, theta),
over several random networks and a finite-difference step sweep. Also:
  * cross-checks the independent forward-mode sensitivity,
  * stresses the transversality assumption by driving a crossing toward tangency,
  * contrasts the exact gradient with a spike-time-frozen gradient to show the
    spike-timing terms carry essentially all the signal for spiking parameters.

PASS criterion: best-h relative error < 1e-3 for W, tau_m, theta on every network.

Run:  python3 run_finite_diff_gate.py
"""

from __future__ import annotations

import numpy as np

from spikeflow.adjoint import backward
from spikeflow.finite_diff import fd_gradient
from spikeflow.forward import simulate
from spikeflow.forward_sensitivity import forward_sensitivity
from spikeflow.gate_metrics import group_vectors, relerr
from spikeflow.params import NetworkParams, init_params
from spikeflow.sampling import sample_flow_matching

H_SWEEP = [1e-3, 1e-4, 1e-5, 1e-6]
GROUPS = ["W", "tau_m", "theta"]
PASS_TOL = 1e-3


def best_h_relerr(p: NetworkParams, u, target, g_adj_groups: dict):
    """For each group, the smallest relative error over the finite-difference h sweep."""
    out = {grp: (np.inf, None) for grp in g_adj_groups}
    for h in H_SWEEP:
        g_fd = group_vectors(fd_gradient(p, u, target, h=h))
        for grp in g_adj_groups:
            e = relerr(g_adj_groups[grp], g_fd[grp])
            if e < out[grp][0]:
                out[grp] = (e, h)
    return out


def frozen_time_gradient(p: NetworkParams, u, fwd, target) -> dict:
    """Gradient if spike times were held fixed (the readout-weight path only).

    V_out(S) = sum_k W_out[:,n_k] exp(-(S-s_k)/tau_out). Freezing s_k, only W_out has
    a nonzero derivative; every parameter that acts solely by moving spike times
    (W_in, W2, tau_m, theta) gets exactly zero. The exact gradient is large for all
    of them -> the spike-time sensitivity terms are the entire signal there.
    """
    g = {"W_in": np.zeros_like(p.W_in), "W2": np.zeros_like(p.W2),
         "W_out": np.zeros_like(p.W_out), "tau_m": 0.0, "theta": 0.0}
    resid2 = 2.0 * (fwd.Vout_S - target)
    for ev in fwd.events:
        if ev.layer == 2:
            g["W_out"][:, ev.n] += resid2 * np.exp(-(p.S - ev.time) / p.tau_out)
    return g


def run_accuracy_gate(seeds=(0, 1, 2, 3, 4)) -> bool:
    print("=" * 74)
    print("ACCURACY GATE  (exact adjoint vs central finite differences, best-h)")
    print("=" * 74)
    print(f"{'seed':>4} | {'W':>12} {'tau_m':>12} {'theta':>12} | {'fwd-mode chk':>14}")
    print("-" * 74)
    worst = {grp: 0.0 for grp in GROUPS}
    for seed in seeds:
        p = init_params(seed=seed)
        s = sample_flow_matching(seed=seed + 100, d=p.m)
        fwd = simulate(p, s.u)
        g_adj = group_vectors(backward(p, s.u, s.target, fwd))
        res = best_h_relerr(p, s.u, s.target, {g: g_adj[g] for g in GROUPS})
        # forward-mode cross-check on a representative scalar (theta) and one weight
        fm_theta = forward_sensitivity(p, s.u, s.target, fwd, ("theta",))
        fm_chk = relerr(fm_theta, float(g_adj["theta"][0]))
        line = f"{seed:>4} | "
        for grp in GROUPS:
            e, h = res[grp]
            worst[grp] = max(worst[grp], e)
            line += f"{e:>12.2e} "
        line += f"| {fm_chk:>14.2e}"
        print(line)
    print("-" * 74)
    ok = all(worst[g] < PASS_TOL for g in GROUPS)
    print("worst-case rel_err:  " + "  ".join(f"{g}={worst[g]:.2e}" for g in GROUPS))
    print(f"PASS criterion (< {PASS_TOL:.0e}): {'PASS' if ok else 'FAIL'}")
    return ok


def run_surrogate_gap(seed=0) -> None:
    print("\n" + "=" * 74)
    print("SPIKE-TIMING SIGNAL  (exact vs spike-time-frozen gradient)")
    print("=" * 74)
    p = init_params(seed=seed)
    s = sample_flow_matching(seed=seed + 100, d=p.m)
    fwd = simulate(p, s.u)
    g_adj = group_vectors(backward(p, s.u, s.target, fwd))
    g_frozen = group_vectors(frozen_time_gradient(p, s.u, fwd, s.target))
    print(f"{'group':>8} | {'gap (frozen vs exact)':>22} | {'|exact|':>12} {'|frozen|':>12}")
    for grp in ["W_out", "W2", "W_in", "tau_m", "theta"]:
        gap = relerr(g_frozen[grp], g_adj[grp])
        print(f"{grp:>8} | {gap:>22.3e} | {np.linalg.norm(g_adj[grp]):>12.4e} "
              f"{np.linalg.norm(g_frozen[grp]):>12.4e}")
    print("Reading: ~0 gap on W_out (enters linearly, no timing); ~1 (100%) on the")
    print("spiking parameters -> their entire gradient is the spike-time sensitivity.")


def run_transversality_stress(seed=0) -> None:
    print("\n" + "=" * 74)
    print("TRANSVERSALITY STRESS  (drive a layer-2 crossing toward tangency, Vdot->0)")
    print("=" * 74)
    print(f"{'w2_scale':>9} | {'min|Vdot|':>10} | {'#L2 spk':>7} | {'rel_err W':>10} "
          f"{'rel_err th':>11} | {'FD stable':>9}")
    s = sample_flow_matching(seed=seed + 100, d=2)
    for w2 in [0.6, 0.4, 0.3, 0.25, 0.22, 0.20, 0.19, 0.185]:
        p = init_params(seed=seed, w2_scale=w2)
        fwd = simulate(p, s.u)
        l2 = [e for e in fwd.events if e.layer == 2]
        if not l2:
            print(f"{w2:>9.3f} | {'--':>10} | {0:>7} | (no layer-2 spikes)")
            continue
        min_vdot = min(abs(e.vdot) for e in l2)
        g_adj = group_vectors(backward(p, s.u, s.target, fwd))
        # FD stability: spike count identical under +/- h?
        from dataclasses import replace
        h = 1e-6
        n0 = len(simulate(p, s.u).events)
        npv = len(simulate(replace(p.copy(), theta=p.theta + h), s.u).events)
        nmv = len(simulate(replace(p.copy(), theta=p.theta - h), s.u).events)
        stable = (npv == n0 == nmv)
        g_fd = group_vectors(fd_gradient(p, s.u, s.target, h=1e-5))
        eW = relerr(g_adj["W"], g_fd["W"])
        eth = relerr(g_adj["theta"], g_fd["theta"])
        print(f"{w2:>9.3f} | {min_vdot:>10.2e} | {len(l2):>7} | {eW:>10.2e} {eth:>11.2e} "
              f"| {'yes' if stable else 'NO':>9}")
    print("Reading: as min|Vdot|->0 the 1/Vdot jump amplifies and FD loses spike-count")
    print("stability (gate unreliable). Stay in the regime where FD is stable.")


if __name__ == "__main__":
    ok = run_accuracy_gate()
    run_surrogate_gap()
    run_transversality_stress()
    print("\n" + "=" * 74)
    print(f"GATE RESULT: {'PASS  -- exact gradient certified by finite differences' if ok else 'FAIL'}")
    print("=" * 74)
