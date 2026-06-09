"""Generation-bound experiment: Wasserstein-2 distance vs spiking resolution T.

Samples the trained spiking velocity field at several time-grid resolutions T and
measures W2(p_hat_spike(T), q). Variance reduction so the O(1/T) coding term is
visible above sampling noise:

  * common noise x0 reused across every T (paired) -> the only quantity that varies
    with T is the spike-time grid quantization, not the random draw;
  * analytic target quantiles for q (a known Gaussian) -> zero target-sampling noise
    (see generation.w2_to_gaussian);
  * a multiprocessing pool over the x0 batch -> large n on many cores (built for the
    server run); n_workers<=1 falls back to a serial loop for a clean local smoke test.

The reported floor a_inf = W2(p_hat_exact, q) is the continuous-time (T=None) sampler;
the trend fit W2(T) ~ a + b/T should give a ~ a_inf, b > 0, and a monotone decrease.
This toy is one-dimensional (the 1-D W2 is a sorted-sample integral); d>1 needs a
multivariate W2 and is out of scope here.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from .generation import sample_pfode, w2_to_gaussian
from .params import NetworkParams

_WORKER_P: NetworkParams | None = None


def _init_worker(p: NetworkParams) -> None:
    """Pool initializer: pin the network in each worker once (not per task)."""
    global _WORKER_P
    _WORKER_P = p


def _sample_worker(args):
    x0, T, n_steps = args
    return sample_pfode(_WORKER_P, x0, T=T, n_steps=n_steps)


def sample_population(p: NetworkParams, x0_batch: np.ndarray, T, n_steps: int = 16,
                      n_workers: int = 1) -> np.ndarray:
    """Roll out the PF-ODE from each x0 row; return generated endpoints, shape [n, d]."""
    if n_workers <= 1:
        return np.array([sample_pfode(p, x0, T=T, n_steps=n_steps) for x0 in x0_batch])
    args = [(x0, T, n_steps) for x0 in x0_batch]
    chunk = max(1, len(args) // (n_workers * 4))
    with mp.Pool(n_workers, initializer=_init_worker, initargs=(p,)) as pool:
        out = pool.map(_sample_worker, args, chunksize=chunk)
    return np.array(out)


def fit_a_plus_b_over_T(Ts, vals):
    """Least-squares fit vals ~ a + b/T; return (a, b, R^2)."""
    x = 1.0 / np.asarray(Ts, dtype=float)
    y = np.asarray(vals, dtype=float)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    yhat = A @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-30
    return float(a), float(b), 1.0 - ss_res / ss_tot


def w2_vs_T(p: NetworkParams, Ts, q_mu: float, q_sigma: float, n_samp: int = 2000,
            n_steps: int = 16, n_workers: int = 1, seed: int = 7, n_rep: int = 1) -> dict:
    """W2(p_hat_spike(T), q) vs T, the exact-sampler floor, the paired quantization
    distance W2(spike(T), exact), and the a + b/T fit.

    q is N(q_mu, q_sigma^2). Within a replicate the x0 noise is shared across all T
    (paired). n_rep independent replicates (different x0 seeds) are averaged: the b/T
    term on the full W2-to-q is small relative to one replicate's sampling noise, so
    averaging over replicates (noise ~ 1/sqrt(n_rep)) is what makes the trend clean.
    Each row carries the per-T standard error across replicates.
    """
    from .generation import w2_1d  # paired quantization distance (training-independent)

    Ts = [int(T) for T in Ts]
    perT_q = {T: [] for T in Ts}    # W2(spike(T), q) per replicate
    perT_quant = {T: [] for T in Ts}  # W2(spike(T), exact) per replicate
    floors, means, stds = [], [], []
    for r in range(n_rep):
        x0 = np.random.default_rng(seed + r).standard_normal((n_samp, p.m))
        gen = {T: sample_population(p, x0, T, n_steps, n_workers) for T in Ts}
        gen_inf = sample_population(p, x0, None, n_steps, n_workers)
        for T in Ts:
            perT_q[T].append(w2_to_gaussian(gen[T], q_mu, q_sigma))
            perT_quant[T].append(w2_1d(gen[T], gen_inf))
        floors.append(w2_to_gaussian(gen_inf, q_mu, q_sigma))
        means.append(float(gen_inf.mean())); stds.append(float(gen_inf.std()))

    rows = [(T, float(np.mean(perT_q[T])), float(np.std(perT_q[T]) / np.sqrt(n_rep)))
            for T in Ts]
    quant = [(T, float(np.mean(perT_quant[T]))) for T in Ts]
    a_inf = float(np.mean(floors))
    a, b, r2 = fit_a_plus_b_over_T([T for T, *_ in rows], [w for _, w, _ in rows])
    mono = all(rows[i][1] >= rows[i + 1][1] - 1e-4 for i in range(len(rows) - 1))
    return {
        "rows": rows,                        # [(T, mean W2(spike_T, q), sem), ...]
        "quant": quant,                      # [(T, W2(spike_T, exact)), ...]  (part B')
        "a_inf": a_inf,                      # W2(exact sampler, q) = empirical floor
        "fit": {"a": a, "b": b, "r2": r2},   # W2(T) ~ a + b/T
        "monotone": mono,
        "gen_mean": float(np.mean(means)),
        "gen_std": float(np.mean(stds)),
        "n_rep": n_rep,
    }
