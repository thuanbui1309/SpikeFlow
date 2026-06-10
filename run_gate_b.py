"""Stage-1 Gate B (dense FC): hidden-count cadence drift of the snapped vs exact sampler.

Thin driver: builds the wide dense FC velocity net and hands it to the shared sweep
engine (`spikeflow.gate_b_sweep`). The conv stage uses the same engine via
`run_gate_b_conv.py`.

The LOCAL SMOKE config (tiny m) exercises the pipeline + verdict logic only; it is NOT
the GO/PIVOT decision. The real verdict is the wide-FC m=3072 multi-T run.

Run:  uv run python run_gate_b.py --quick                                    # smoke (tiny net)
      uv run python run_gate_b.py --net fc --n-workers 32 --probe            # cost/ETA only
      uv run python run_gate_b.py --net fc --n-workers 32 \
                                  --save results/a2_trajectory_violation.npz  # full wide-FC
      # full-run defaults: n1=n2=32, m=3072, T sweep 16..1024, K 50/100, n_samples 16
"""

from __future__ import annotations

import argparse

from spikeflow import gate_b_sweep as gb
from spikeflow.fc_params import init_fc_params


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", type=str, default="fc", choices=["fc"],
                    help="network topology (conv has its own driver)")
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
    ap.add_argument("--probe", action="store_true",
                    help="time ONE simulate + ONE worst-case trajectory, print an ETA, "
                         "then exit -- a fast 'is it hung or just slow?' diagnostic")
    return ap.parse_args()


def main() -> None:
    a = build_args()
    if a.quick:                                  # one-flag smoke: tiny net + short sweep
        a.n1, a.n2, a.m = 16, 16, 1
        a.T, a.K, a.n_samples = [16, 64], [50], 6

    print("=== SpikeFlow Stage-1 Gate B (dense FC): hidden-count cadence drift ===\n")

    guard_ok = gb.self_check_cache_guard()
    print(f"cache-guard tripwire live = {guard_ok}")
    if not guard_ok:
        raise SystemExit("ABORT: cross-step cache-guard did not fire; tripwire is not "
                         "active, so a stale snapped ForwardResult could be reused.")

    p = init_fc_params(seed=a.seed, n1=a.n1, n2=a.n2, m=a.m)
    print(f"net={a.net}  n1={a.n1} n2={a.n2} m={a.m}  "
          f"T={a.T}  K={a.K}  n_samples={a.n_samples}  workers={a.n_workers}\n")

    if a.probe:
        gb.run_probe(p, a.T, a.K, a.n_samples, a.n_workers, a.bisect_iters, a.sample_seed)
        return

    rows, curves, paired_max = gb.run_sweep(p, a.T, a.K, a.n_samples, a.n_workers,
                                            a.bisect_iters, a.sample_seed)

    print(f"\npaired-control max flips (MUST be 0) = {paired_max}")
    if paired_max != 0:
        print("  *** INVALID RUN: paired control non-zero -> harness reads counts wrong ***")

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
        gb.save_gate_b(
            a.save, a.T, a.K, rows, curves, paired_max,
            net_config=[a.n1, a.n2, a.m, a.n_samples],
            title="SpikeFlow Stage-1 Gate B (dense FC): hidden-count cadence drift",
            header_line=f"net={a.net} n1={a.n1} n2={a.n2} m={a.m} n_samples={a.n_samples}",
            scope=scope, guard_ok=guard_ok,
        )


if __name__ == "__main__":
    main()
