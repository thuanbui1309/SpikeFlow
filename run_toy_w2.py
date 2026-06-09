"""Generation-bound headline: Wasserstein-2 of the spiking sampler vs resolution T.

Trains the spiking velocity field on a 1-D Gaussian target with the certified exact
gradient, then samples with the spike-resolution sampler at several T. Three read-outs:

  (A) velocity coding error eps_quant(T) = mean||v^T - v^inf||  ->  expect O(1/T)
      (a sanity check that an O(1/T) spike-TIME coding term exists; this is the generic
      rate of grid-sampling a smooth readout, not a measurement of any specific constant);
  (B') quantization-only distance W2(p_hat_spike(T), p_hat_exact)  [the headline trend:
      training-independent because both samplers use the same trained net and shared
      noise, so it isolates the O(1/T) spike-time term from the training floor; note it
      measures convergence to the *exact (continuous-time) sampler*, not to q];
  (B) W2(p_hat_spike(T), q) vs T  [floor-dominated: the spike effect on the distance to
      q is second-order ~ W2(spike,exact)^2/(2*floor), so the O(1/T) term is below
      sampling noise here -- reported for honesty, not as a clean a+b/T fit].

Variance reduction: shared x0 across T + analytic target quantiles + averaging over
n_rep replicates make the small b/T term visible above estimator noise. The default
network is the lowest-floor regime found by run_config_sweep.py (more neurons fit the
target better); override the regime via the CLI to explore.

Run:  uv run python run_toy_w2.py --quick                                   # smoke test
      uv run python run_toy_w2.py --n-workers 64 --n-samp 4000 --n-rep 8 --n-iter 1500 \
                                  --save results/headline.npz
"""

from __future__ import annotations

import argparse

import numpy as np

from spikeflow.generation import velocity_quant_error
from spikeflow.params import init_params
from spikeflow.theorem2 import fit_a_plus_b_over_T, w2_vs_T
from spikeflow.train import train

Q_MU, Q_SIGMA = 1.6, 0.6


def q_sampler(rng, n):
    """1-D target: a moderately-separated unimodal Gaussian N(Q_MU, Q_SIGMA^2)."""
    return Q_MU + Q_SIGMA * rng.standard_normal((n, 1))


def build_args():
    ap = argparse.ArgumentParser()
    # training
    ap.add_argument("--n-iter", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    # evaluation
    ap.add_argument("--n-samp", type=int, default=2000)
    ap.add_argument("--n-steps", type=int, default=16)
    ap.add_argument("--n-rep", type=int, default=4, help="x0-seed replicates, averaged")
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--quick", action="store_true", help="tiny config for a smoke test")
    # network regime (defaults = lowest-floor 'bigger-net' from the config sweep)
    ap.add_argument("--n1", type=int, default=12)
    ap.add_argument("--n2", type=int, default=12)
    ap.add_argument("--window", type=float, default=14.0)
    ap.add_argument("--tau-out", type=float, default=8.0)
    ap.add_argument("--bias", type=float, default=1.25)
    ap.add_argument("--w-in", type=float, default=0.8)
    return ap.parse_args()


def main():
    a = build_args()
    if a.quick:
        a.n_iter, a.n_samp, a.n_steps, a.n_rep = 120, 200, 8, 2
    p = init_params(seed=3, d=1, n1=a.n1, n2=a.n2, window=a.window, tau_out=a.tau_out,
                    bias_level=a.bias, w_in_scale=a.w_in, w2_scale=0.5, w_out_scale=0.4)

    print(f"Training (exact-gradient Adam, n_iter={a.n_iter}, net {a.n1}/{a.n2})...")
    p, hist = train(p, q_sampler, n_iter=a.n_iter, batch=a.batch, lr=a.lr, seed=a.seed)
    print(f"final val loss ~ {hist['val'][-1][1]:.4f}\n")

    Ts = [16, 32, 64, 128, 256] if not a.quick else [16, 64]

    # (A) velocity coding error vs T (no ODE rollout: cheap, robust to training quality)
    rng = np.random.default_rng(7)
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

    # (B) + (B') via the shared, replicate-averaged pipeline
    res = w2_vs_T(p, Ts, Q_MU, Q_SIGMA, n_samp=a.n_samp, n_steps=a.n_steps,
                  n_workers=a.n_workers, seed=7, n_rep=a.n_rep)

    print("(B') W2(p_hat_spike(T), p_hat_exact)  [training-independent]:")
    aq, bq, r2q = fit_a_plus_b_over_T([T for T, _ in res["quant"]], [w for _, w in res["quant"]])
    for T, w in res["quant"]:
        print(f"   T={T:>4}  W2={w:.5e}  W2*T={w*T:.4f}")
    print(f"   fit ~ a+b/T: a={aq:.3e} b={bq:.3e} R^2={r2q:.4f}\n")

    print(f"(B) HEADLINE  W2(p_hat_spike(T), q) vs T   [mean over {res['n_rep']} reps +/- sem]:")
    for T, w, sem in res["rows"]:
        print(f"   T={T:>4}  W2={w:.5f} +/- {sem:.5f}")
    print(f"   T=inf  W2={res['a_inf']:.5f}   (exact-sampler floor)")
    print(f"   gen(inf): mean={res['gen_mean']:+.3f} std={res['gen_std']:.3f} "
          f"(target mean {Q_MU:+.2f} std {Q_SIGMA:.2f})")
    f = res["fit"]
    print(f"   fit W2 ~ a+b/T: a={f['a']:.4f} b={f['b']:.4f} R^2={f['r2']:.4f}  "
          f"monotone={'yes' if res['monotone'] else 'no'}  (want b>0, a~floor {res['a_inf']:.4f})")

    if a.save:
        import os
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        np.savez(a.save, Ts=[T for T, _, _ in res["rows"]], w2=[w for _, w, _ in res["rows"]],
                 sem=[s for _, _, s in res["rows"]], a_inf=res["a_inf"],
                 fit=[f["a"], f["b"], f["r2"]], gen_mean=res["gen_mean"], gen_std=res["gen_std"],
                 eps_T=Ts_a, eps=eps, quant=[w for _, w in res["quant"]])
        print(f"\nsaved headline data -> {a.save}")


if __name__ == "__main__":
    main()
