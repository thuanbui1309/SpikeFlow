"""Adam training of the spiking velocity field on the flow-matching loss.

Uses the exact backward-adjoint gradient (the certified one). Only the weights are
updated; tau_m and theta are held fixed (matching the certified gradient scope).
v_theta is regressed onto the conditional flow-matching velocity x1 - x0 at (x_t, t).

Stability recipe (the spiking loss surface is rough — a weight step that pushes a
membrane just over threshold adds a whole spike, a discontinuity in the readout):
  * warmup + cosine learning-rate schedule — small early steps avoid spike-count
    jumps before momentum settles, cosine decay anneals into a flat minimum;
  * decoupled weight decay — without it the optimiser can grow the input drive
    without bound, shrinking the inter-spike interval until the event-driven sim
    slows to a crawl and the loss roughens; decay keeps the drive (hence spike
    count) bounded;
  * gradient-norm clipping — caps the rare huge step near a near-tangent crossing
    (the 1/V_dot saltation factor);
  * a fixed held-out batch tracks validation loss and the mean spike count, so a
    runaway spike count is visible rather than silent.
"""

from __future__ import annotations

import numpy as np

from .adjoint import backward
from .forward import simulate
from .params import NetworkParams

WEIGHT_KEYS = ("W_in", "W2", "W_out")


def _lr_at(it: int, n_iter: int, lr: float, lr_min: float, warmup: int) -> float:
    """Linear warmup over `warmup` iters, then cosine decay lr -> lr_min."""
    if it <= warmup:
        return lr * it / max(1, warmup)
    prog = (it - warmup) / max(1, n_iter - warmup)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + np.cos(np.pi * min(1.0, prog)))


def _clip(grads: dict, max_norm: float) -> None:
    total = np.sqrt(sum(float(np.sum(grads[k] ** 2)) for k in WEIGHT_KEYS))
    if total > max_norm:
        for k in WEIGHT_KEYS:
            grads[k] *= max_norm / total


def _batch_loss_and_spikes(p: NetworkParams, batch) -> tuple[float, float, float]:
    """Mean loss and mean per-eval layer-1/layer-2 spike counts over a fixed batch."""
    loss = n1 = n2 = 0.0
    for u, tgt in batch:
        fwd = simulate(p, u)
        loss += fwd.loss(tgt)
        n1 += sum(1 for e in fwd.events if e.layer == 1)
        n2 += sum(1 for e in fwd.events if e.layer == 2)
    k = len(batch)
    return loss / k, n1 / k, n2 / k


def _draw_batch(rng, q_sampler, batch: int, d: int):
    """One flow-matching batch: list of (u=(x_t,t), target=x1-x0)."""
    x1b = q_sampler(rng, batch)
    x0b = rng.standard_normal((batch, d))
    tb = rng.uniform(0.0, 1.0, size=batch)
    out = []
    for j in range(batch):
        xt = (1.0 - tb[j]) * x0b[j] + tb[j] * x1b[j]
        out.append((np.concatenate([xt, [tb[j]]]), x1b[j] - x0b[j]))
    return out


def train(p: NetworkParams, q_sampler, n_iter=800, batch=16, lr=2e-3, lr_min=1e-4,
          warmup_frac=0.05, weight_decay=2e-3, seed=0, clip=20.0, val_batch=48,
          val_every=50, log_every=100, k_sub=16):
    """Train weights with Adam + schedule + decoupled weight decay.

    q_sampler(rng, n) -> array of data samples x1, shape [n, d]. k_sub sets the
    readout-integral quadrature density (only the input-weight gradient uses it;
    synaptic/readout-weight gradients are exact event sums), so a modest value
    suffices. Returns (p, history) where history has keys 'train' (per-iter loss),
    'val' / 'spikes' (lists of (iter, ...)).
    """
    rng = np.random.default_rng(seed)
    d = p.m
    val = _draw_batch(np.random.default_rng(seed + 10_000), q_sampler, val_batch, d)
    warmup = max(1, int(warmup_frac * n_iter))
    mom = {k: np.zeros_like(getattr(p, k)) for k in WEIGHT_KEYS}
    vel = {k: np.zeros_like(getattr(p, k)) for k in WEIGHT_KEYS}
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = {"train": [], "val": [], "spikes": []}
    for it in range(1, n_iter + 1):
        cur_lr = _lr_at(it, n_iter, lr, lr_min, warmup)
        gsum = {k: np.zeros_like(getattr(p, k)) for k in WEIGHT_KEYS}
        lsum = 0.0
        for u, tgt in _draw_batch(rng, q_sampler, batch, d):
            fwd = simulate(p, u)
            lsum += fwd.loss(tgt)
            g = backward(p, u, tgt, fwd, k_sub=k_sub)
            for k in WEIGHT_KEYS:
                gsum[k] += g[k]
        for k in WEIGHT_KEYS:
            gsum[k] /= batch
        _clip(gsum, clip)
        for k in WEIGHT_KEYS:
            mom[k] = b1 * mom[k] + (1 - b1) * gsum[k]
            vel[k] = b2 * vel[k] + (1 - b2) * gsum[k] ** 2
            mhat = mom[k] / (1 - b1 ** it)
            vhat = vel[k] / (1 - b2 ** it)
            arr = getattr(p, k)
            arr -= cur_lr * (mhat / (np.sqrt(vhat) + eps) + weight_decay * arr)
        hist["train"].append(lsum / batch)
        if val_every and (it % val_every == 0 or it == n_iter):
            vl, vn1, vn2 = _batch_loss_and_spikes(p, val)
            hist["val"].append((it, vl))
            hist["spikes"].append((it, vn1, vn2))
            if log_every and (it % log_every == 0 or it == n_iter):
                print(f"  iter {it:4d}  lr {cur_lr:.2e}  train {np.mean(hist['train'][-val_every:]):.4f}"
                      f"  val {vl:.4f}  spikes(n1/n2) {vn1:.1f}/{vn2:.1f}")
    return p, hist
