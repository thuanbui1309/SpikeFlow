"""Generation-bound toy: Wasserstein-2 of the spiking sampler vs resolution T.

Trains the spiking velocity field on a 1-D Gaussian target with the certified exact
gradient, then samples with the spike-resolution sampler at several T. Three read-outs:

  (A) velocity coding error eps_quant(T) = mean||v^T - v^inf||  ->  expect O(1/T)
      (the deterministic grid-quantization lemma; robust to training quality);
  (B') quantization-only distance W2(p_hat_spike(T), p_hat_exact)  [training-independent,
      eps_train cancels because both use the same trained net and shared noise];
  (B) headline: W2(p_hat_spike(T), q) vs T, fit a + b/T  (monotone decrease at rate
      1/T toward the training/expressivity floor a = W2(exact sampler, q)).

Variance reduction (shared x0 across T + analytic target quantiles) makes the b/T term
visible above estimator noise. Scale n_samp / n_workers up on a many-core server.

Run:  uv run python run_toy_w2.py                 # default
      uv run python run_toy_w2.py --quick         # fast local smoke test
      uv run python run_toy_w2.py --n-workers 32 --n-samp 4000 --save results/headline.npz
"""

from __future__ import annotations

import argparse

import numpy as np

from spikeflow.generation import sample_pfode, velocity_quant_error, w2_1d, w2_to_gaussian
from spikeflow.params import init_params
from spikeflow.theorem2 import fit_a_plus_b_over_T, sample_population
from spikeflow.train import train

Q_MU, Q_SIGMA = 1.6, 0.6


def q_sampler(rng, n):
    """1-D target: a moderately-separated unimodal Gaussian N(Q_MU, Q_SIGMA^2)."""
    return Q_MU + Q_SIGMA * rng.standard_normal((n, 1))


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=800)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n-samp", type=int, default=2000)
    ap.add_argument("--n-steps", type=int, default=16)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--quick", action="store_true", help="tiny config for a local smoke test")
    return ap.parse_args()


def main():
    a = build_args()
    if a.quick:
        a.n_iter, a.n_samp, a.n_steps = 120, 200, 8
    # Input-sensitive, spike-count-bounded regime: low bias so the input (not a large
    # constant drive) modulates the firing pattern, which keeps the velocity field
    # input-dependent and the event count small (fast, smooth loss surface).
    p = init_params(seed=3, d=1, n1=8, n2=8, window=14.0, tau_out=8.0,
                    bias_level=1.25, w_in_scale=0.8, w2_scale=0.5, w_out_scale=0.4)

    print(f"Training (exact-gradient Adam, n_iter={a.n_iter})...")
    p, hist = train(p, q_sampler, n_iter=a.n_iter, batch=a.batch, lr=a.lr, seed=a.seed)
    print(f"final val loss ~ {hist['val'][-1][1]:.4f}\n")

    Ts = [16, 32, 64, 128, 256] if not a.quick else [16, 64]
    rng = np.random.default_rng(7)

    # (A) velocity coding error vs T (no ODE rollout: cheap, robust to training)
    test_u = []
    for _ in range(150 if not a.quick else 40):
        x0 = rng.standard_normal(1); x1 = q_sampler(rng, 1)[0]; t = rng.uniform()
        test_u.append(np.concatenate([(1 - t) * x0 + t * x1, [t]]))
    Ts_a = [8, 16, 32, 64, 128, 256, 512] if not a.quick else [16, 64]
    eps = [velocity_quant_error(p, np.array(test_u), T) for T in Ts_a]
    aa, ba, r2a = fit_a_plus_b_over_T(Ts_a, eps)
    print("(A) velocity coding error eps_quant(T):")
    for T, e in zip(Ts_a, eps):
        print(f"   T={T:>4}  eps={e:.5e}  eps*T={e*T:.4f}")
    print(f"   fit eps ~ a+b/T: a={aa:.3e} b={ba:.3e} R^2={r2a:.4f}\n")

    # sample once (shared x0) -> reuse for (B') and (B)
    x0s = np.random.default_rng(7).standard_normal((a.n_samp, 1))
    gen = {T: sample_population(p, x0s, T, a.n_steps, a.n_workers) for T in Ts}
    gen_inf = sample_population(p, x0s, None, a.n_steps, a.n_workers)

    # (B') quantization-only distance W2(spike(T), exact) [eps_train cancels]
    bp = [(T, w2_1d(gen[T], gen_inf)) for T in Ts]
    ab, bb, r2b = fit_a_plus_b_over_T([T for T, _ in bp], [w for _, w in bp])
    print("(B') W2(p_hat_spike(T), p_hat_exact)  [training-independent]:")
    for T, w in bp:
        print(f"   T={T:>4}  W2={w:.5e}  W2*T={w*T:.4f}")
    print(f"   fit ~ a+b/T: a={ab:.3e} b={bb:.3e} R^2={r2b:.4f}\n")

    # (B) headline W2(spike(T), q) vs T
    rows = [(T, w2_to_gaussian(gen[T], Q_MU, Q_SIGMA)) for T in Ts]
    a_inf = w2_to_gaussian(gen_inf, Q_MU, Q_SIGMA)
    aB, bB, r2B = fit_a_plus_b_over_T([T for T, _ in rows], [w for _, w in rows])
    mono = all(rows[i][1] >= rows[i + 1][1] - 1e-4 for i in range(len(rows) - 1))
    print("(B) HEADLINE  W2(p_hat_spike(T), q) vs T:")
    for T, w in rows:
        print(f"   T={T:>4}  W2={w:.5f}")
    print(f"   T=inf  W2={a_inf:.5f}   (exact-sampler floor)")
    print(f"   gen(inf): mean={gen_inf.mean():+.3f} std={gen_inf.std():.3f} "
          f"(target mean {Q_MU:+.2f} std {Q_SIGMA:.2f})")
    print(f"   fit W2 ~ a+b/T: a={aB:.4f} b={bB:.4f} R^2={r2B:.4f}  "
          f"monotone={'yes' if mono else 'no'}  (want b>0, a~floor {a_inf:.4f})")

    if a.save:
        import os
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        np.savez(a.save, Ts=[T for T, _ in rows], w2=[w for _, w in rows], a_inf=a_inf,
                 fit=[aB, bB, r2B], eps_T=Ts_a, eps=eps, bprime=[w for _, w in bp])
        print(f"\nsaved headline data -> {a.save}")


if __name__ == "__main__":
    main()
