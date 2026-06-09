"""Torch-path finite-difference gate for the exact spiking velocity-field gradient.

Mirrors run_finite_diff_gate.py exactly (same seeds, same sample offset, same
H sweep, same groups, same PASS_TOL), but the gradient under test comes from the
torch.autograd.Function wrapper (ExactEventAdjoint) instead of the NumPy adjoint
called directly. The ORACLE is still fd_gradient (NumPy central differences); we
certify that running the verified adjoint through the torch autograd boundary --
float64 end-to-end -- reproduces the finite-difference gradient group-by-group.

Also asserts the torch forward Vout matches NumPy simulate().Vout_S to <=1e-10
(same closed-form computation, just routed through the autograd Function).

PASS criterion: best-h relative error < 1e-3 for W, tau_m, theta on every network.

Run:  uv run python run_torch_gate.py
"""

from __future__ import annotations

import numpy as np
import torch

from spikeflow.finite_diff import fd_gradient
from spikeflow.forward import simulate
from spikeflow.gate_metrics import group_vectors, relerr
from spikeflow.params import init_params
from spikeflow.sampling import sample_flow_matching
from spikeflow.torch_core import ExactEventAdjoint

H_SWEEP = [1e-3, 1e-4, 1e-5, 1e-6]
GROUPS = ["W", "tau_m", "theta"]
PASS_TOL = 1e-3


def best_h_relerr(p, u, target, g_adj_groups: dict) -> dict:
    """For each group, the smallest relative error over the finite-difference h sweep."""
    out = {grp: (np.inf, None) for grp in g_adj_groups}
    for h in H_SWEEP:
        g_fd = group_vectors(fd_gradient(p, u, target, h=h))
        for grp in g_adj_groups:
            e = relerr(g_adj_groups[grp], g_fd[grp])
            if e < out[grp][0]:
                out[grp] = (e, h)
    return out


def _leaf_tensors(p):
    """Build float64 leaf tensors for the trained set; bias as a frozen constant."""
    w_in = torch.tensor(p.W_in, dtype=torch.float64, requires_grad=True)
    w2 = torch.tensor(p.W2, dtype=torch.float64, requires_grad=True)
    w_out = torch.tensor(p.W_out, dtype=torch.float64, requires_grad=True)
    tau_m = torch.tensor(float(p.tau_m), dtype=torch.float64, requires_grad=True)
    theta = torch.tensor(float(p.theta), dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(p.bias, dtype=torch.float64, requires_grad=False)
    return w_in, bias, w2, w_out, tau_m, theta


def torch_grad_dict(p, u, target) -> tuple[dict, np.ndarray]:
    """Run the torch autograd path; return the adjoint grad dict + the torch Vout."""
    w_in, bias, w2, w_out, tau_m, theta = _leaf_tensors(p)
    fixed = {"tau_s": p.tau_s, "tau_out": p.tau_out, "S": p.S, "v_rest": p.v_rest}
    u_t = torch.tensor(u, dtype=torch.float64)
    target_t = torch.tensor(target, dtype=torch.float64)

    vout = ExactEventAdjoint.apply(w_in, bias, w2, w_out, tau_m, theta, u_t, target_t, fixed)
    # Squared residual, NO 0.5 factor -> dL/dv = 2(v-target) matches the adjoint seed.
    loss = ((vout - target_t) ** 2).sum()
    loss.backward()

    grad = {
        "W_in": w_in.grad.detach().cpu().double().numpy(),
        "W2": w2.grad.detach().cpu().double().numpy(),
        "W_out": w_out.grad.detach().cpu().double().numpy(),
        "tau_m": float(tau_m.grad.item()),
        "theta": float(theta.grad.item()),
    }
    return grad, vout.detach().cpu().double().numpy()


def run_accuracy_gate(seeds=(0, 1, 2, 3, 4)) -> bool:
    print("=" * 74)
    print("TORCH GATE  (autograd-wrapped exact adjoint vs central finite differences)")
    print("=" * 74)
    print(f"{'seed':>4} | {'W':>12} {'tau_m':>12} {'theta':>12} | {'fwd match':>12}")
    print("-" * 74)
    worst = {grp: 0.0 for grp in GROUPS}
    fwd_ok = True
    for seed in seeds:
        p = init_params(seed=seed)
        s = sample_flow_matching(seed=seed + 100, d=p.m)
        ref = simulate(p, s.u)  # NumPy reference forward

        grad, vout_torch = torch_grad_dict(p, s.u, s.target)
        fwd_err = float(np.linalg.norm(vout_torch - ref.Vout_S))
        fwd_ok = fwd_ok and (fwd_err <= 1e-10)

        g_adj = group_vectors(grad)
        res = best_h_relerr(p, s.u, s.target, {g: g_adj[g] for g in GROUPS})
        line = f"{seed:>4} | "
        for grp in GROUPS:
            e, _h = res[grp]
            worst[grp] = max(worst[grp], e)
            line += f"{e:>12.2e} "
        line += f"| {fwd_err:>12.2e}"
        print(line)
    print("-" * 74)
    ok = all(worst[g] < PASS_TOL for g in GROUPS)
    print("worst-case rel_err:  " + "  ".join(f"{g}={worst[g]:.2e}" for g in GROUPS))
    if not fwd_ok:
        print("FORWARD MISMATCH: torch Vout deviates from NumPy simulate() by > 1e-10")
    print(f"PASS criterion (< {PASS_TOL:.0e}): {'PASS' if (ok and fwd_ok) else 'FAIL'}")
    return ok and fwd_ok


if __name__ == "__main__":
    ok = run_accuracy_gate()
    print("\n" + "=" * 74)
    msg = ("PASS  -- torch autograd path certified by finite differences"
           if ok else "FAIL")
    print(f"TORCH GATE RESULT: {msg}")
    print("=" * 74)
