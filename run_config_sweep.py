"""Parallel configuration search for the generation-bound toy.

Each configuration (network regime + training schedule) is trained and evaluated in
its own worker process, so a many-core server sweeps the whole grid at once. For each
config it reports the training/expressivity floor a_inf = W2(exact sampler, q), the
fitted 1/T slope b, the fit R^2, monotonicity, and the generated mean/std. The goal is
to pick the regime with the smallest floor whose W2-vs-T trend is clean (b>0, monotone,
high R^2) for the headline figure.

Cross-config parallelism only: the inner sampler runs serially inside each worker, so
do not also raise its worker count. Edit GRID to explore; --quick shrinks everything.

Run:  uv run python run_config_sweep.py --n-workers 8
      uv run python run_config_sweep.py --quick           # local smoke test
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from spikeflow.params import init_params
from spikeflow.theorem2 import w2_vs_T
from spikeflow.train import train

Q_MU, Q_SIGMA = 1.6, 0.6

# Network-regime + training knobs. bias_level near theta keeps the input (not a large
# constant drive) in control of firing -> input-dependent velocity + bounded spike count.
GRID = [
    {"tag": "h2-base",     "n1": 8,  "n2": 8,  "window": 14.0, "tau_out": 8.0, "bias": 1.25, "win": 0.8, "lr": 2e-3, "iter": 800,  "wd": 2e-3},
    {"tag": "more-iter",   "n1": 8,  "n2": 8,  "window": 14.0, "tau_out": 8.0, "bias": 1.25, "win": 0.8, "lr": 2e-3, "iter": 1500, "wd": 2e-3},
    {"tag": "bigger-net",  "n1": 12, "n2": 12, "window": 14.0, "tau_out": 8.0, "bias": 1.25, "win": 0.8, "lr": 2e-3, "iter": 1000, "wd": 2e-3},
    {"tag": "lower-bias",  "n1": 8,  "n2": 8,  "window": 14.0, "tau_out": 8.0, "bias": 1.10, "win": 0.9, "lr": 2e-3, "iter": 1000, "wd": 2e-3},
    {"tag": "high-win",    "n1": 8,  "n2": 8,  "window": 14.0, "tau_out": 8.0, "bias": 1.25, "win": 1.1, "lr": 2e-3, "iter": 1000, "wd": 2e-3},
    {"tag": "long-window", "n1": 10, "n2": 10, "window": 18.0, "tau_out": 10.0,"bias": 1.20, "win": 0.9, "lr": 2e-3, "iter": 1000, "wd": 2e-3},
    {"tag": "low-lr-long", "n1": 8,  "n2": 8,  "window": 14.0, "tau_out": 8.0, "bias": 1.25, "win": 0.8, "lr": 1e-3, "iter": 1800, "wd": 1e-3},
    {"tag": "more-decay",  "n1": 8,  "n2": 8,  "window": 14.0, "tau_out": 8.0, "bias": 1.25, "win": 0.8, "lr": 3e-3, "iter": 1000, "wd": 5e-3},
]


def q_sampler(rng, n):
    return Q_MU + Q_SIGMA * rng.standard_normal((n, 1))


def _eval_config(cfg: dict) -> dict:
    """Train + evaluate one config (serial inside; parallel across configs)."""
    t0 = time.time()
    p = init_params(seed=3, d=1, n1=cfg["n1"], n2=cfg["n2"], window=cfg["window"],
                    tau_out=cfg["tau_out"], bias_level=cfg["bias"], w_in_scale=cfg["win"],
                    w2_scale=0.5, w_out_scale=0.4)
    p, hist = train(p, q_sampler, n_iter=cfg["iter"], batch=cfg["batch"], lr=cfg["lr"],
                    weight_decay=cfg["wd"], seed=1, log_every=0, val_every=max(50, cfg["iter"]))
    res = w2_vs_T(p, cfg["Ts"], Q_MU, Q_SIGMA, n_samp=cfg["n_samp"],
                  n_steps=cfg["n_steps"], n_workers=1, seed=7)
    return {"tag": cfg["tag"], "val_loss": hist["val"][-1][1], "a_inf": res["a_inf"],
            "b": res["fit"]["b"], "r2": res["fit"]["r2"], "monotone": res["monotone"],
            "gen_mean": res["gen_mean"], "gen_std": res["gen_std"],
            "spikes": hist["spikes"][-1][1:], "secs": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=min(8, mp.cpu_count()))
    ap.add_argument("--n-samp", type=int, default=600)
    ap.add_argument("--n-steps", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--save", type=str, default="results/config_sweep.json")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    grid = [dict(c, batch=a.batch, n_samp=a.n_samp, n_steps=a.n_steps,
                 Ts=[24, 64, 192]) for c in GRID]
    if a.quick:
        grid = [dict(c, iter=120, n_samp=150, n_steps=8, Ts=[16, 64]) for c in grid[:3]]

    print(f"sweeping {len(grid)} configs on {a.n_workers} workers...")
    t0 = time.time()
    with mp.Pool(a.n_workers) as pool:
        results = pool.map(_eval_config, grid)
    results.sort(key=lambda r: r["a_inf"])  # smaller floor first

    print(f"\ndone in {time.time()-t0:.0f}s. ranked by floor a_inf = W2(exact, q):\n")
    print(f"{'tag':>12} {'a_inf':>7} {'val':>7} {'b':>8} {'R2':>6} {'mono':>5} "
          f"{'genμ':>6} {'genσ':>6} {'n1/n2':>9} {'s':>5}")
    for r in results:
        n1, n2 = r["spikes"]
        print(f"{r['tag']:>12} {r['a_inf']:>7.4f} {r['val_loss']:>7.3f} {r['b']:>8.4f} "
              f"{r['r2']:>6.3f} {str(r['monotone']):>5} {r['gen_mean']:>+6.2f} "
              f"{r['gen_std']:>6.2f} {n1:>4.0f}/{n2:<4.0f} {r['secs']:>5.0f}")

    if a.save:
        import os
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        with open(a.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved -> {a.save}\nbest floor: {results[0]['tag']} (a_inf={results[0]['a_inf']:.4f})")


if __name__ == "__main__":
    main()
