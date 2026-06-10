"""Stage-1 Gate B: hidden-spike-COUNT cadence drift of the snapped vs exact sampler.

For each (T, K) it rolls n_samples paired Heun trajectories (one exact T=None, one
snapped T) from shared x0 seeds and measures the FREE-RUN violation_rate (fraction of
(sample, step) pairs where the integer total hidden count differs between the two
state paths) plus the cumulative-flip curve. The paired-same-state rate is a sanity
control that must be 0 (counts are T-independent at fixed input).

Verdict per (T, K):
  GO       : free-run rate <= 0.05 at K=50 AND cumulative flips stay <= 0.05*(k+1)
             for all k (no super-linear compounding) -> cadence stable.
  PIVOT    : free-run rate >= 0.20 at K=50 OR super-linear cumulative growth
             (final cumulative >> 0.05*K) -> drift feeds drift.
  BORDERLINE: rate in (0.05, 0.20) -> report, do not auto-decide.

The LOCAL SMOKE config (tiny m) exercises the pipeline + verdict logic only; it is
NOT the GO/PIVOT decision. The real verdict is the wide-FC m=3072 multi-T run.

Run:  uv run python run_gate_b.py --quick                                   # smoke (tiny net)
      uv run python run_gate_b.py --net fc --n-workers 32 \
                                  --save results/a2_trajectory_violation.npz  # full wide-FC
      # full-run defaults: n1=n2=32, m=3072, T sweep 16..1024, K 50/100, n_samples 16
"""

from __future__ import annotations

import argparse

import numpy as np

from spikeflow import cache_guard
from spikeflow.fc_params import init_fc_params
from spikeflow.trajectory_metrics import aggregate_violation

GO_RATE = 0.05
PIVOT_RATE = 0.20
DECISION_K = 50


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", type=str, default="fc", choices=["fc"],
                    help="network topology (conv added later)")
    ap.add_argument("--seed", type=int, default=0, help="network init seed")
    ap.add_argument("--sample-seed", type=int, default=100, help="x0 batch seed")
    ap.add_argument("--n1", type=int, default=32)
    ap.add_argument("--n2", type=int, default=32)
    ap.add_argument("--m", type=int, default=3072,
                    help="velocity dim (wide-FC gate=3072; --quick sets the tiny smoke net)")
    ap.add_argument("--T", type=int, nargs="+", default=[16, 32, 64, 128, 256, 1024],
                    help="snap-grid resolutions")
    ap.add_argument("--K", type=int, nargs="+", default=[50, 100], help="Heun step counts")
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--bisect-iters", type=int, default=25)
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--quick", action="store_true", help="tiny config for a smoke test")
    return ap.parse_args()


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


def main() -> None:
    a = build_args()
    if a.quick:                                  # one-flag smoke: tiny net + short sweep
        a.n1, a.n2, a.m = 16, 16, 1
        a.T, a.K, a.n_samples = [16, 64], [50], 6

    print("=== SpikeFlow Stage-1 Gate B: hidden-count cadence drift ===\n")

    guard_ok = self_check_cache_guard()
    print(f"cache-guard tripwire live = {guard_ok}")
    if not guard_ok:
        raise SystemExit("ABORT: cross-step cache-guard did not fire; tripwire is not "
                         "active, so a stale snapped ForwardResult could be reused.")

    p = init_fc_params(seed=a.seed, n1=a.n1, n2=a.n2, m=a.m)
    print(f"net={a.net}  n1={a.n1} n2={a.n2} m={a.m}  "
          f"T={a.T}  K={a.K}  n_samples={a.n_samples}  workers={a.n_workers}\n")

    rows, curves = [], {}
    paired_max_all = 0
    for K in a.K:
        for T in a.T:
            agg = aggregate_violation(p, T, K, a.n_samples, seed=a.sample_seed,
                                      n_workers=a.n_workers, bisect_iters=a.bisect_iters)
            cum = agg["cumulative_flip_curve"]
            curves[(T, K)] = cum
            paired_max_all = max(paired_max_all, agg["paired_control_max"])
            vd = verdict_for(agg["violation_rate"] if K == DECISION_K else None, cum)
            rows.append((T, K, agg["violation_rate"], agg["paired_control_max"],
                         float(cum[-1]), vd))

    print(f"{'T':>6} {'K':>5} {'free_rate':>11} {'paired':>7} {'cum_flips':>11}  verdict")
    print("-" * 64)
    for T, K, rate, pm, cum_end, vd in rows:
        print(f"{T:>6} {K:>5} {rate:>11.4f} {pm:>7} {cum_end:>11.3f}  {vd}")

    print(f"\npaired-control max flips (MUST be 0) = {paired_max_all}")
    if paired_max_all != 0:
        print("  *** INVALID RUN: paired control non-zero -> harness reads counts wrong ***")

    # The scope marker must survive into the saved report: the report file is what
    # gets read on the other side of the push/pull loop, and a smoke-scale run is
    # indistinguishable from the decision run by its table alone.
    if a.m < 256:
        scope = ("SMOKE config (tiny m) -- validates pipeline + verdict logic ONLY, "
                 "NOT the GO/PIVOT decision")
    else:
        scope = (f"DECISION RUN: wide-FC m={a.m} n1={a.n1} n2={a.n2} -- this IS the "
                 "Stage-1 GO/PIVOT measurement (read the K=50 rows)")
    print(f"\n{scope}")
    if a.m < 256:
        print("Run the wide-FC net (defaults: --n1 32 --n2 32 --m 3072) for the real "
              "Stage-1 verdict.")

    if a.save:
        import os
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        save = {f"cum_T{T}_K{K}": cum for (T, K), cum in curves.items()}
        save["Ts"] = np.array(a.T)
        save["Ks"] = np.array(a.K)
        save["rate_table"] = np.array([(T, K, r) for T, K, r, *_ in rows], dtype=float)
        save["paired_max"] = paired_max_all
        save["net_config"] = np.array([a.n1, a.n2, a.m, a.n_samples])
        np.savez(a.save, **save)
        report = a.save.rsplit(".", 1)[0] + "_report.txt"
        with open(report, "w") as f:
            f.write("SpikeFlow Stage-1 Gate B: hidden-count cadence drift\n")
            f.write(f"net={a.net} n1={a.n1} n2={a.n2} m={a.m} n_samples={a.n_samples}\n")
            f.write(f"scope: {scope}\n")
            f.write(f"cache-guard live={guard_ok}  paired_control_max={paired_max_all}\n\n")
            f.write(f"{'T':>6} {'K':>5} {'free_rate':>11} {'cum_flips':>11}  verdict\n")
            for T, K, rate, pm, cum_end, vd in rows:
                f.write(f"{T:>6} {K:>5} {rate:>11.4f} {cum_end:>11.3f}  {vd}\n")
        print(f"\nsaved curves -> {a.save}\nsaved report -> {report}")


if __name__ == "__main__":
    main()
