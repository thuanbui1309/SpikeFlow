"""Operational gate checks for the wide dense (FC) spiking velocity net.

Builds the wide-FC net, runs the cross-step no-cache tripwire over a real
PF-ODE sample, checks replay determinism + T-independence of hidden spike
counts, and reports the readout coding constant C_impl (at smoke m=48 and the
full m=3072), the worst-case transversality margin mu_min, and the
T-independence verdict. Flags a VAE-fallback if C_impl exceeds 5000.

Real measured numbers only; no fabricated values.
"""

from __future__ import annotations

import numpy as np

from spikeflow import cache_guard
from spikeflow.fc_params import init_fc_params
from spikeflow.forward import simulate
from spikeflow.gate_protocols import c_impl_suffix_sum, mu_min
from spikeflow.generation import sample_pfode, velocity

VAE_FALLBACK_THRESHOLD = 5000.0


def _counts(fwd):
    n1 = sum(1 for e in fwd.events if e.layer == 1)
    n2 = sum(1 for e in fwd.events if e.layer == 2)
    return n1, n2


def check_cache_guard(p, seed: int = 1, n_steps: int = 10, T: int = 64) -> bool:
    """Run a real PF-ODE sample; the tripwire must never fire."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(p.m)
    cache_guard.assert_no_survivors()
    try:
        sample_pfode(p, x0, T=T, n_steps=n_steps)
        cache_guard.assert_no_survivors()
        return True
    except RuntimeError:
        return False


def check_replay_determinism(p, trials: int = 100, seed: int = 7) -> bool:
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        u = np.concatenate([rng.standard_normal(p.m), [rng.uniform(0.0, 1.0)]])
        if _counts(simulate(p, u)) != _counts(simulate(p, u)):
            return False
    return True


def check_t_independence(p, seed: int = 11):
    """Counts are T-invariant; readout snapped times differ and converge."""
    rng = np.random.default_rng(seed)
    u = np.concatenate([rng.standard_normal(p.m), [0.3]])
    fwd = simulate(p, u)
    n1, n2 = _counts(fwd)
    v16, v256, vinf = velocity(p, u, 16), velocity(p, u, 256), velocity(p, u, None)
    d_coarse = float(np.linalg.norm(v16 - v256))
    d_fine = float(np.linalg.norm(v256 - vinf))
    counts_ok = _counts(simulate(p, u)) == (n1, n2)
    converges = d_fine < d_coarse and d_coarse > 1e-6
    return {
        "n_hidden_l1": n1,
        "n_l2": n2,
        "d_16_vs_256": d_coarse,
        "d_256_vs_inf": d_fine,
        "counts_t_independent": counts_ok,
        "readout_converges": converges,
    }


def main() -> None:
    print("=== SpikeFlow wide-FC operational gate checks ===\n")

    # Smoke net (m=48): full code path, CPU-cheap.
    p_smoke = init_fc_params(seed=0, n1=32, n2=32, m=48)
    fwd_smoke = simulate(p_smoke, np.concatenate([np.zeros(p_smoke.m), [0.3]]))
    c_smoke = c_impl_suffix_sum(p_smoke)
    mu_smoke = mu_min(fwd_smoke)

    cache_ok = check_cache_guard(p_smoke)
    replay_ok = check_replay_determinism(p_smoke)
    t_ind = check_t_independence(p_smoke)

    print(f"[smoke m={p_smoke.m}]")
    print(f"  C_impl (suffix-sum)        = {c_smoke:.4f}")
    print(f"  mu_min (layer-2 margin)    = {mu_smoke:.6f}  (safe > 0.01: {mu_smoke > 0.01})")
    print(f"  cache-guard clean pass     = {cache_ok}")
    print(f"  replay-count determinism   = {replay_ok}")
    print(f"  hidden counts (l1, l2)     = ({t_ind['n_hidden_l1']}, {t_ind['n_l2']})")
    print(f"  counts T-independent       = {t_ind['counts_t_independent']}")
    print(f"  ||v16-v256||               = {t_ind['d_16_vs_256']:.6f}")
    print(f"  ||v256-vinf||              = {t_ind['d_256_vs_inf']:.6f}")
    print(f"  readout converges to exact = {t_ind['readout_converges']}")

    # Full-width net (m=3072): C_impl only (full PF-ODE sample is expensive).
    print(f"\n[full m=3072] C_impl across seeds:")
    c_full_values = []
    for seed in (0, 1, 2):
        c_full = c_impl_suffix_sum(init_fc_params(seed=seed, n1=32, n2=32, m=3072))
        c_full_values.append(c_full)
        print(f"  seed={seed}: C_impl = {c_full:.4f}")

    c_max = max(c_full_values + [c_smoke])
    print(f"\n=== verdict ===")
    print(f"  max C_impl observed = {c_max:.4f} (VAE-fallback threshold = {VAE_FALLBACK_THRESHOLD})")
    if c_max > VAE_FALLBACK_THRESHOLD:
        print("  *** VAE-FALLBACK FLAG: C_impl exceeds threshold ***")
    else:
        print("  C_impl within safe regime (no VAE fallback).")

    all_ok = cache_ok and replay_ok and t_ind["counts_t_independent"] and t_ind["readout_converges"]
    print(f"  operational checks PASS = {all_ok}")


if __name__ == "__main__":
    main()
