"""Torch autograd boundary around the verified NumPy event-driven adjoint.

This wraps the pure-NumPy forward simulation and exact backward adjoint in a
torch.autograd.Function so the network weights become torch leaf tensors and the
finite-difference gate can run through the torch path. It WRAPS the oracle, it
does not reimplement it: forward() calls simulate(), backward() calls the exact
adjoint backward(). Both stay float64 end-to-end -- float32 cannot meet the 1e-3
gate tolerance, so every tensor is cast to double before crossing into NumPy.

torch is imported only here (and torch_module.py); the package __init__ never
imports torch, so `import spikeflow` still works without torch installed.
"""

from __future__ import annotations

import numpy as np
import torch

from . import adjoint
from .forward import simulate
from .params import NetworkParams


def params_from_tensors(w_in, bias, w2, w_out, tau_m, theta, fixed: dict) -> NetworkParams:
    """Build a NetworkParams from torch tensors + a dict of fixed scalar config.

    Arrays are detached/moved to CPU and cast to float64 NumPy (the simulation is
    closed-form in double; the gate compares against the float64 fd oracle).
    Scalars tau_m/theta are taken as Python floats. `fixed` carries the config
    fields that are never differentiated (tau_s, tau_out, S, v_rest).
    """
    return NetworkParams(
        W_in=w_in.detach().cpu().double().numpy(),
        bias=bias.detach().cpu().double().numpy(),
        W2=w2.detach().cpu().double().numpy(),
        W_out=w_out.detach().cpu().double().numpy(),
        tau_m=float(tau_m.item()),
        theta=float(theta.item()),
        tau_s=float(fixed["tau_s"]),
        tau_out=float(fixed["tau_out"]),
        S=float(fixed["S"]),
        v_rest=float(fixed.get("v_rest", 0.0)),
    )


class ExactEventAdjoint(torch.autograd.Function):
    """Autograd node: forward = event-driven simulate, backward = exact adjoint.

    The squared-residual flow-matching loss ||v - target||^2 has dL/dv = 2(v-target).
    The NumPy adjoint bakes that exact seed in (lam_Vout(S) = 2*(Vout-target)), so
    when the torch-side loss is the same squared residual the upstream grad_output
    arriving here equals that hardcoded seed -- this Function returns Vout_S and
    lets backward() reuse the seed directly, reusing the oracle UNCHANGED.

    Event detection (the layer-2 bisection) lives entirely inside the NumPy call and
    is never differentiated: the spike COUNT/order is piecewise-constant in the
    parameters, so it has zero gradient a.e.; the smooth dependence on parameters
    flows through the spike TIMES, which the analytic adjoint handles via the
    transposed saltation jump. Differentiating the bracketing loop would be wrong.
    """

    @staticmethod
    def forward(ctx, w_in, bias, w2, w_out, tau_m, theta, u, target, fixed):
        p = params_from_tensors(w_in, bias, w2, w_out, tau_m, theta, fixed)
        u_np = u.detach().cpu().double().numpy()
        fwd = simulate(p, u_np)
        # Stash plain (non-tensor) objects for backward; fwd is independent of target.
        ctx.p = p
        ctx.u_np = u_np
        ctx.fwd = fwd
        ctx.target_np = (
            None if target is None else target.detach().cpu().double().numpy()
        )
        out_device = w_in.device
        return torch.from_numpy(fwd.Vout_S.copy()).to(device=out_device, dtype=torch.float64)

    @staticmethod
    def backward(ctx, grad_out):
        # The oracle adjoint needs the same residual seed the loss implies:
        # 2*(Vout - target') == grad_out  ->  target' = Vout - grad_out/2.
        # That reconstructs the exact seed for ANY upstream grad_out while reusing
        # the ctx-saved forward trajectory (which does not depend on target).
        grad_np = grad_out.detach().cpu().double().numpy()
        target_eff = ctx.fwd.Vout_S - 0.5 * grad_np
        g = adjoint.backward(ctx.p, ctx.u_np, target_eff, ctx.fwd, k_sub=64)

        device = grad_out.device
        w_in_g = torch.from_numpy(g["W_in"].copy()).to(device=device, dtype=torch.float64)
        w2_g = torch.from_numpy(g["W2"].copy()).to(device=device, dtype=torch.float64)
        w_out_g = torch.from_numpy(g["W_out"].copy()).to(device=device, dtype=torch.float64)
        tau_m_g = torch.tensor(g["tau_m"], device=device, dtype=torch.float64)
        theta_g = torch.tensor(g["theta"], device=device, dtype=torch.float64)

        # bias is FROZEN: the adjoint exposes no dL/dbias and train.py trains only
        # {W_in, W2, W_out}. Return None rather than fabricate a bias gradient.
        # u, target, fixed are non-differentiable -> None.
        return (w_in_g, None, w2_g, w_out_g, tau_m_g, theta_g, None, None, None)
