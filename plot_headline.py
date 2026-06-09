"""Render the generation-bound figure from a saved headline run.

Reads the .npz produced by run_toy_w2.py --save and draws the three panels that carry
the Theorem-2 story:
  (a) velocity coding error eps_quant(T)         -- log-log, ~1/T reference slope
  (b) sampler-to-exact distance W2(spike(T),exact) -- the training-independent b/T signal
  (c) generation distance W2(spike(T), q) +/- sem  -- with the a + b/T fit and floor a

Publication defaults: vector PDF + 300-dpi PNG, >=8 pt fonts, a colour-blind-safe
palette, honest axes, and a self-contained title per panel.

Run:  uv sync --extra plot
      uv run python plot_headline.py results/headline.npz --out results/headline
"""

from __future__ import annotations

import argparse

import numpy as np

# Colour-blind-safe (Wong 2011): blue, orange, green.
C_DATA, C_FIT, C_FLOOR = "#0072B2", "#D55E00", "#009E73"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default="results/headline")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})

    d = np.load(args.npz)
    Ts = d["Ts"].astype(float)
    w2 = d["w2"]; sem = d["sem"] if "sem" in d else np.zeros_like(w2)
    a_inf = float(d["a_inf"]); fa, fb, fr2 = d["fit"]
    eps_T = d["eps_T"].astype(float); eps = d["eps"]
    quant = d["quant"]

    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.6), constrained_layout=True)

    # (a) velocity coding error, log-log with a 1/T guide
    ax[0].loglog(eps_T, eps, "o", color=C_DATA, ms=5, label="measured")
    guide = eps[0] * eps_T[0] / eps_T
    ax[0].loglog(eps_T, guide, "--", color=C_FIT, lw=1.3, label=r"$\propto 1/T$")
    ax[0].set_title("(a) velocity coding error")
    ax[0].set_xlabel("resolution $T$"); ax[0].set_ylabel(r"$\epsilon_{\mathrm{quant}}(T)$")
    ax[0].legend(frameon=False)

    # (b) sampler-to-exact distance, log-log with b/T fit
    bq = np.polyfit(1.0 / Ts, quant, 1)  # quant ~ a + b/T
    ax[1].loglog(Ts, quant, "s", color=C_DATA, ms=5, label="measured")
    grid = np.linspace(Ts.min(), Ts.max(), 100)
    ax[1].loglog(grid, max(bq[1], 1e-9) + bq[0] / grid, "--", color=C_FIT, lw=1.3,
                 label=r"$a+b/T$")
    ax[1].set_title("(b) sampler $\\to$ exact")
    ax[1].set_xlabel("resolution $T$"); ax[1].set_ylabel(r"$W_2(\hat p_T,\hat p_\infty)$")
    ax[1].legend(frameon=False)

    # (c) generation distance to target with a + b/T fit and the floor a
    ax[2].errorbar(Ts, w2, yerr=sem, fmt="o", color=C_DATA, ms=5, capsize=3, label="measured")
    grid = np.linspace(Ts.min(), Ts.max(), 100)
    ax[2].plot(grid, fa + fb / grid, "--", color=C_FIT, lw=1.3,
               label=fr"$a+b/T$ ($R^2$={fr2:.2f})")
    ax[2].axhline(a_inf, color=C_FLOOR, lw=1.2, ls=":", label=fr"floor $a={a_inf:.3f}$")
    ax[2].set_title("(c) distance to target")
    ax[2].set_xlabel("resolution $T$"); ax[2].set_ylabel(r"$W_2(\hat p_T, q)$")
    ax[2].legend(frameon=False)

    for p in (f"{args.out}.pdf", f"{args.out}.png"):
        fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"saved {args.out}.pdf and {args.out}.png  (b={fb:.4f}, R^2={fr2:.3f}, floor={a_inf:.3f})")


if __name__ == "__main__":
    main()
