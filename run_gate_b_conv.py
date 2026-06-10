"""Stage-2 Gate: conv event-identity + Gate B cadence drift on the small conv net.

The conv velocity net (weight sharing + pooling, materialised as a structured dense
net in `conv_params`) plugs into the SAME event engine, exact adjoint, readout-snap
sampler, and Gate B sweep engine as the dense FC stage. This driver adds only the two
conv-specific measurements before the trajectory sweep:

  1. spike-count sanity  -- one simulate at full scale, abort if the count explodes
     (the fan-in lesson: a mis-scaled drive makes the event sim intractable).
  2. event-identity audit -- replay determinism, hidden-count T-invariance, and a
     tie/degenerate-crossing audit (does weight sharing keep counts exact?).

Then it recomputes C_impl (readout coding constant) + mu_min (transversality) at conv
scale, re-arms the VAE fallback flag at >5000, and runs Gate B via the shared engine.

Run:  uv run python run_gate_b_conv.py --quick                               # tiny smoke
      uv run python run_gate_b_conv.py --n-workers 32 --probe                # cost/ETA only
      uv run python run_gate_b_conv.py --n-workers 32 \
                                       --save results/gate_b_conv.npz         # full conv gate
"""

from __future__ import annotations

import argparse

import numpy as np

from spikeflow import gate_b_sweep as gb
from spikeflow.conv_event_identity import audit_event_identity, min_crossing_gap, tie_stress
from spikeflow.conv_params import init_conv_params
from spikeflow.forward import simulate
from spikeflow.gate_protocols import c_impl_suffix_sum, mu_min

VAE_THRESHOLD = 5000.0


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-seed", type=int, default=100)
    ap.add_argument("--img", type=int, nargs=3, default=[3, 32, 32],
                    help="input image C H W (= velocity shape)")
    ap.add_argument("--c1", type=int, default=4)
    ap.add_argument("--k1", type=int, default=4)
    ap.add_argument("--stride1", type=int, default=4)
    ap.add_argument("--c2", type=int, default=4)
    ap.add_argument("--k2", type=int, default=2)
    ap.add_argument("--stride2", type=int, default=2)
    ap.add_argument("--T", type=int, nargs="+", default=[16, 32, 64, 128, 256, 1024])
    ap.add_argument("--K", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--bisect-iters", type=int, default=25)
    ap.add_argument("--audit-n", type=int, default=24, help="inputs for the event-identity audit")
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--quick", action="store_true", help="tiny conv for a smoke test")
    ap.add_argument("--probe", action="store_true",
                    help="time one simulate + one worst-case trajectory + print ETA, then exit")
    return ap.parse_args()


def run_event_identity(p, sample_seed, audit_n, bisect_iters):
    """Audit + worst-case mu_min over the audit batch; returns (audit dict, mu_worst)."""
    rng = np.random.default_rng(sample_seed + 1)
    inputs = np.concatenate(
        [rng.standard_normal((audit_n, p.m)), rng.random((audit_n, 1))], axis=1)
    audit = audit_event_identity(p, inputs, Ts=(16, 256), bisect_iters=bisect_iters)
    mu_worst = min(mu_min(simulate(p, u, bisect_iters=bisect_iters)) for u in inputs)
    return audit, float(mu_worst)


def main() -> None:
    a = build_args()
    if a.quick:                                  # tiny conv: small image + short sweep
        a.img = [3, 8, 8]
        a.c1, a.k1, a.stride1 = 2, 4, 4
        a.c2, a.k2, a.stride2 = 2, 2, 2
        a.T, a.K, a.n_samples, a.audit_n = [16, 64], [50], 6, 8

    print("=== SpikeFlow Stage-2 Gate: conv event-identity + Gate B cadence drift ===\n")

    guard_ok = gb.self_check_cache_guard()
    print(f"cache-guard tripwire live = {guard_ok}")
    if not guard_ok:
        raise SystemExit("ABORT: cross-step cache-guard did not fire; tripwire inactive.")

    p, spec = init_conv_params(
        seed=a.seed, img=tuple(a.img),
        c1=a.c1, k1=a.k1, stride1=a.stride1, c2=a.c2, k2=a.k2, stride2=a.stride2)
    print(f"conv: img={tuple(a.img)} l1={spec.l1} l2={spec.l2}  "
          f"N1={p.N1} N2={p.N2} m={p.m}  k1={a.k1}/s{a.stride1} k2={a.k2}/s{a.stride2}")
    print(f"sweep: T={a.T} K={a.K} n_samples={a.n_samples} workers={a.n_workers}\n")

    # 1. Spike-count sanity at full scale (the fan-in lesson) -- abort if intractable.
    rng = np.random.default_rng(a.sample_seed)
    u0 = np.concatenate([rng.standard_normal(p.m), [0.0]])
    fwd0 = simulate(p, u0, bisect_iters=a.bisect_iters)
    n1_0 = sum(1 for e in fwd0.events if e.layer == 1)
    n2_0 = sum(1 for e in fwd0.events if e.layer == 2)
    print(f"spike-count sanity: n1={n1_0} n2={n2_0} total={n1_0 + n2_0}")
    if n1_0 + n2_0 > 50000:
        raise SystemExit("ABORT: conv spike count >50k -- drive over-scaled (fan-in); "
                         "the event sim is intractable. Reduce channels or re-check scaling.")
    if n1_0 == 0 or n2_0 == 0:
        raise SystemExit("ABORT: a conv hidden layer is silent -- gate would pass vacuously.")

    # 2. Conv event-identity audit (the Stage-2-specific question).
    audit, mu_worst = run_event_identity(p, a.sample_seed, a.audit_n, a.bisect_iters)
    ident_ok = audit["replay_stable"] and audit["counts_t_invariant"]
    print(f"event-identity: replay_stable={audit['replay_stable']} "
          f"t_invariant={audit['counts_t_invariant']} "
          f"min_gap={audit['global_min_gap']:.2e} near_tie_rate={audit['near_tie_rate']:.3f} "
          f"(mean n1={audit['mean_n1']:.0f} n2={audit['mean_n2']:.0f}, N={audit['n_inputs']})")

    # 2b. Tie stress: uniform images force weight-sharing into the near-tie regime that
    # random inputs never reach, then confirm counts stay deterministic + order-benign.
    ts = tie_stress(p, img=tuple(a.img), bisect_iters=a.bisect_iters)
    snap_grid = p.S / max(a.T)                                  # finest readout snap step
    ident_ok = ident_ok and ts["tie_replay_stable"]            # deterministic tie-break is required
    print(f"tie-stress: tie_min_gap={ts['tie_min_gap']:.2e} (snap grid S/Tmax={snap_grid:.2e}; "
          f"gap<<grid => ties below readout resolution)  replay_stable={ts['tie_replay_stable']} "
          f"perturb_count_change={ts['perturb_count_change_rate']:.3f}")

    # 3. Readout coding constant + transversality at conv scale.
    c_impl = c_impl_suffix_sum(p)
    vae_flag = c_impl > VAE_THRESHOLD
    print(f"C_impl={c_impl:.1f} (VAE fallback flag={vae_flag}, threshold {VAE_THRESHOLD:.0f})  "
          f"mu_min={mu_worst:.4f}")

    ident_report = [
        f"conv: img={tuple(a.img)} l1={spec.l1} l2={spec.l2} N1={p.N1} N2={p.N2} m={p.m}",
        f"spike-count sanity n1={n1_0} n2={n2_0} total={n1_0 + n2_0}",
        f"event-identity replay_stable={audit['replay_stable']} "
        f"counts_t_invariant={audit['counts_t_invariant']} "
        f"global_min_gap={audit['global_min_gap']:.3e} near_tie_rate={audit['near_tie_rate']:.3f}",
        f"tie-stress tie_min_gap={ts['tie_min_gap']:.3e} snap_grid={snap_grid:.3e} "
        f"replay_stable={ts['tie_replay_stable']} perturb_count_change={ts['perturb_count_change_rate']:.3f}",
        f"C_impl={c_impl:.1f} VAE_flag={vae_flag} mu_min={mu_worst:.4f}",
    ]

    if a.probe:
        gb.run_probe(p, a.T, a.K, a.n_samples, a.n_workers, a.bisect_iters, a.sample_seed)
        return

    if not ident_ok:
        print("\n*** event-identity FAILED (replay or T-invariance) -- conv counts are not "
              "deterministic/T-stable. This is a Stage-2 red flag; surfacing, not auto-pivoting. ***")

    rows, curves, paired_max = gb.run_sweep(p, a.T, a.K, a.n_samples, a.n_workers,
                                            a.bisect_iters, a.sample_seed)

    print(f"\npaired-control max flips (MUST be 0) = {paired_max}")
    if paired_max != 0:
        print("  *** INVALID RUN: paired control non-zero -> harness reads counts wrong ***")

    is_decision = tuple(a.img) == (3, 32, 32)
    if is_decision:
        scope = (f"DECISION RUN: conv img={tuple(a.img)} N1={p.N1} N2={p.N2} m={p.m} -- this "
                 "IS the Stage-2 GO/PIVOT measurement (read K=50 rows + event-identity above)")
    else:
        scope = ("SMOKE config (tiny img) -- validates conv pipeline + verdict logic ONLY, "
                 "NOT the Stage-2 decision")
    print(f"\n{scope}")
    if not is_decision:
        print("Run the CIFAR-scale conv (--img 3 32 32) for the real Stage-2 verdict.")

    if a.save:
        gb.save_gate_b(
            a.save, a.T, a.K, rows, curves, paired_max,
            net_config=[p.N1, p.N2, p.m, a.n_samples],
            title="SpikeFlow Stage-2 Gate: conv event-identity + Gate B cadence drift",
            header_line=f"net=conv img={tuple(a.img)} N1={p.N1} N2={p.N2} m={p.m} "
                        f"n_samples={a.n_samples}",
            scope=scope, guard_ok=guard_ok, extra_lines=ident_report)


if __name__ == "__main__":
    main()
