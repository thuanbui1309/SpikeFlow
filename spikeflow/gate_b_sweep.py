"""Shared Gate B sweep engine: hidden-count cadence drift over the PF-ODE trajectory.

Net-agnostic. Both the dense FC driver (Stage 1) and the conv driver (Stage 2) build
their own NetworkParams and call this engine, so the verdict logic, the live progress
display, the cost probe, and the save format are defined once.

For each (T, K) the engine rolls n_samples paired Heun trajectories (one exact T=None,
one snapped T) from shared x0 seeds and measures the FREE-RUN violation_rate (fraction
of (sample, step) pairs whose integer hidden count differs between the two state paths)
plus the cumulative-flip curve. paired-same-state flips must be 0 (counts are
T-independent at a fixed input).

Verdict per (T, K):
  GO        : free-run rate <= 0.05 at K=50 AND cumulative flips <= 0.05*(k+1) for all
              k (no super-linear compounding) -> cadence stable.
  PIVOT     : free-run rate >= 0.20 at K=50 OR super-linear cumulative growth.
  BORDERLINE: rate in (0.05, 0.20) -> report, do not auto-decide.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from . import cache_guard
from .forward import simulate
from .trajectory_metrics import a2_violation_over_trajectory, aggregate_violation

GO_RATE = 0.05
PIVOT_RATE = 0.20
DECISION_K = 50


def self_check_cache_guard() -> bool:
    """Prove the cross-step tripwire is live: a held dummy must trip the assert."""
    class _Dummy:
        pass

    cache_guard.assert_no_survivors()           # start clean
    held = _Dummy()
    cache_guard.register(held)                  # strong ref held across the boundary
    try:
        cache_guard.assert_no_survivors()
        return False                            # should have raised
    except RuntimeError:
        pass
    del held
    cache_guard.assert_no_survivors()           # clears -> clean again
    return True


def verdict_for(rate_at_k50: float | None, cum_curve: np.ndarray) -> str:
    """GO / PIVOT / BORDERLINE from the K=50 free-run rate + cumulative growth shape."""
    k = len(cum_curve)
    linear_bound = GO_RATE * (np.arange(k) + 1)
    superlinear = cum_curve[-1] > 2.0 * GO_RATE * k          # final >> linear envelope
    within_linear = bool(np.all(cum_curve <= linear_bound + 1e-9))
    if rate_at_k50 is None:                                  # K!=50: shape-only signal
        return "PIVOT" if superlinear else "GO (shape only; decision K=50 not run)"
    if rate_at_k50 >= PIVOT_RATE or superlinear:
        return "PIVOT"
    if rate_at_k50 <= GO_RATE and within_linear:
        return "GO"
    return "BORDERLINE"


def run_probe(p, Ts, Ks, n_samples, n_workers, bisect_iters, sample_seed) -> None:
    """Cost diagnostic: time one simulate + one worst-case trajectory, estimate the sweep.

    Answers 'is the full run hung or just slow?' without committing to the multi-hour
    sweep. Trajectory cost is ~T-independent (T only sets the readout snap grid, not the
    event count), so timing the largest K at one T scales to every config by K.
    """
    rng = np.random.default_rng(sample_seed)
    x0 = rng.standard_normal(p.m)
    u0 = np.concatenate([x0, [0.0]])

    t0 = time.perf_counter()
    fwd = simulate(p, u0, bisect_iters=bisect_iters)
    sim_s = time.perf_counter() - t0
    n1 = sum(1 for e in fwd.events if e.layer == 1)
    n2 = sum(1 for e in fwd.events if e.layer == 2)
    print(f"[probe] one simulate at x0,t=0: {sim_s*1e3:.1f} ms  "
          f"events n1={n1} n2={n2} total={n1+n2}")
    if n1 + n2 > 20000:
        print("  *** WARNING: spike count >20k -- drive likely over-scaled (check w_in "
              "fan-in); each simulate is very expensive and the sweep will crawl ***")

    K_max, T_max = max(Ks), max(Ts)
    print(f"[probe] timing ONE worst-case trajectory T={T_max} K={K_max} "
          f"(4 simulate/Heun step)...", flush=True)
    t0 = time.perf_counter()
    a2_violation_over_trajectory(p, x0, T_max, K_max, bisect_iters)
    traj_s = time.perf_counter() - t0
    print(f"[probe] one trajectory: {traj_s:.1f} s")

    n_eff = min(n_workers, n_samples)
    waves = -(-n_samples // n_eff)                          # ceil(n_samples / n_eff)
    total_s = sum(traj_s * (K / K_max) * waves * len(Ts) for K in Ks)
    print(f"[probe] ESTIMATE full sweep ({len(Ks)*len(Ts)} configs, n_samples="
          f"{n_samples}, n_eff_workers={n_eff}): ~{total_s/60:.1f} min "
          f"(~{total_s/3600:.2f} h)")
    print("[probe] (K=50 decision rows finish in roughly the first "
          f"{total_s/60 * (sum(K for K in Ks if K <= DECISION_K)/sum(Ks)):.1f} min)")
    print("\n[probe] done -- this was a diagnostic only, NO sweep was run. "
          "Drop --probe to run the real sweep.")


def run_sweep(p, Ts, Ks, n_samples, n_workers, bisect_iters, sample_seed):
    """Run the full (K outer, T inner) sweep with live per-config rows + heartbeat.

    Returns (rows, curves, paired_max). Each row = (T, K, rate, paired_max, cum_end,
    verdict). K=DECISION_K rows run (and print) first so the verdict is readable before
    the shape-only K rows finish.
    """
    n_configs = len(Ks) * len(Ts)
    print(f"sweeping {n_configs} configs (K={DECISION_K} decision rows print FIRST); "
          "each row appears as it finishes -- silence between rows = a config in flight,\n"
          "NOT a hang (large nets spend minutes per config; run --probe first for an ETA).\n")
    print(f"{'T':>6} {'K':>5} {'free_rate':>11} {'paired':>7} {'cum_flips':>11} "
          f"{'wall_s':>8}  verdict")
    print("-" * 78)

    rows, curves = [], {}
    paired_max_all = 0
    cfg_i = 0
    for K in Ks:
        for T in Ts:
            cfg_i += 1
            # In-place \r heartbeat only on a real terminal; piped/redirected logs (the
            # ones pasted back over the push/pull loop) get one clean line per sample.
            tty = sys.stderr.isatty()

            def tick(done, total, _T=T, _K=K, _i=cfg_i):
                if tty:
                    print(f"\r  [{_i}/{n_configs}] T={_T:>4} K={_K:>3}  "
                          f"sample {done}/{total} done...", end="", file=sys.stderr, flush=True)
                elif done == total:
                    print(f"  [{_i}/{n_configs}] T={_T} K={_K}  {total} samples done",
                          file=sys.stderr, flush=True)

            cfg_t0 = time.perf_counter()
            agg = aggregate_violation(p, T, K, n_samples, seed=sample_seed,
                                      n_workers=n_workers, bisect_iters=bisect_iters,
                                      progress=tick)
            cfg_wall = time.perf_counter() - cfg_t0
            if tty:
                print("\r" + " " * 60 + "\r", end="", file=sys.stderr, flush=True)
            cum = agg["cumulative_flip_curve"]
            curves[(T, K)] = cum
            paired_max_all = max(paired_max_all, agg["paired_control_max"])
            vd = verdict_for(agg["violation_rate"] if K == DECISION_K else None, cum)
            rows.append((T, K, agg["violation_rate"], agg["paired_control_max"],
                         float(cum[-1]), vd))
            print(f"{T:>6} {K:>5} {agg['violation_rate']:>11.4f} "
                  f"{agg['paired_control_max']:>7} {float(cum[-1]):>11.3f} "
                  f"{cfg_wall:>8.1f}  {vd}", flush=True)

    print("-" * 78)
    return rows, curves, paired_max_all


def save_gate_b(save_path, Ts, Ks, rows, curves, paired_max, net_config,
                title, header_line, scope, guard_ok, extra_lines=()) -> None:
    """Persist curves to npz + a human report carrying the scope marker.

    title/header_line/scope/net_config are net-specific; the table format is shared.
    extra_lines: net-specific report rows (e.g. C_impl, mu_min, event-identity summary).
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    save = {f"cum_T{T}_K{K}": cum for (T, K), cum in curves.items()}
    save["Ts"] = np.array(Ts)
    save["Ks"] = np.array(Ks)
    save["rate_table"] = np.array([(T, K, r) for T, K, r, *_ in rows], dtype=float)
    save["paired_max"] = paired_max
    save["net_config"] = np.array(net_config)
    np.savez(save_path, **save)
    report = save_path.rsplit(".", 1)[0] + "_report.txt"
    with open(report, "w") as f:
        f.write(title + "\n")
        f.write(header_line + "\n")
        f.write(f"scope: {scope}\n")
        f.write(f"cache-guard live={guard_ok}  paired_control_max={paired_max}\n")
        for line in extra_lines:
            f.write(line + "\n")
        f.write("\n")
        f.write(f"{'T':>6} {'K':>5} {'free_rate':>11} {'cum_flips':>11}  verdict\n")
        for T, K, rate, pm, cum_end, vd in rows:
            f.write(f"{T:>6} {K:>5} {rate:>11.4f} {cum_end:>11.3f}  {vd}\n")
    print(f"\nsaved curves -> {save_path}\nsaved report -> {report}")
