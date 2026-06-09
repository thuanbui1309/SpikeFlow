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
            n_steps: int = 16, n_workers: int = 1, seed: int = 7) -> dict:
    """W2(p_hat_spike(T), q) for each T plus the exact-sampler floor and a+b/T fit.

    q is N(q_mu, q_sigma^2). x0 noise is drawn once and shared across all T.
    """
    rng = np.random.default_rng(seed)
    x0_batch = rng.standard_normal((n_samp, p.m))
    rows = []
    for T in Ts:
        gen = sample_population(p, x0_batch, T, n_steps, n_workers)
        rows.append((int(T), w2_to_gaussian(gen, q_mu, q_sigma)))
    gen_inf = sample_population(p, x0_batch, None, n_steps, n_workers)
    a_inf = w2_to_gaussian(gen_inf, q_mu, q_sigma)
    a, b, r2 = fit_a_plus_b_over_T([T for T, _ in rows], [w for _, w in rows])
    mono = all(rows[i][1] >= rows[i + 1][1] - 1e-4 for i in range(len(rows) - 1))
    return {
        "rows": rows,                       # [(T, W2(spike_T, q)), ...]
        "a_inf": a_inf,                      # W2(exact sampler, q) = empirical floor
        "fit": {"a": a, "b": b, "r2": r2},   # W2(T) ~ a + b/T
        "monotone": mono,
        "gen_mean": float(gen_inf.mean()),
        "gen_std": float(gen_inf.std()),
    }
