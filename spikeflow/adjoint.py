"""Exact backward-adjoint gradient (continuous-time EventProp), no surrogate.

A single backward sweep over the recorded trajectory yields the exact gradient of
the simulation-free flow-matching loss w.r.t. {W, tau_m, theta}. The co-state is
propagated in closed form between spikes, undergoes the transposed saltation jump
at each spike (the implicit-function-theorem contribution of the moving spike
time), and is seeded at the window end by the flow-matching residual itself:
lambda_out(S) = 2 (V_out(S) - Delta). Derivation: adjoint-math-spec.md.

Accumulation:
  * W_out, W2, theta  -> exact event sums (no quadrature).
  * tau_m, W_in       -> event-free integrals of lambda . (df/dtheta), evaluated
                         by fine trapezoid on the closed-form state and co-state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forward import ForwardResult, State, membrane_vdot, propagate_state
from .params import NetworkParams


@dataclass
class AdjState:
    lam_V1: np.ndarray
    lam_V2: np.ndarray
    lam_I2: np.ndarray
    lam_Vout: np.ndarray

    def copy(self) -> "AdjState":
        return AdjState(self.lam_V1.copy(), self.lam_V2.copy(),
                        self.lam_I2.copy(), self.lam_Vout.copy())


def propagate_adjoint(lam: AdjState, dt: float, p: NetworkParams) -> AdjState:
    """Advance the co-state backward in time by dt (closed form, transpose ODE).

    lambda_V, lambda_Vout decay along their own time constants; lambda_I picks up
    a coupling term from lambda_V (the I->V coupling, transposed).
    """
    if dt <= 0.0:
        return lam.copy()
    em = np.exp(-dt / p.tau_m)
    es = np.exp(-dt / p.tau_s)
    eo = np.exp(-dt / p.tau_out)
    c = 1.0 / p.tau_m - 1.0 / p.tau_s
    lam_I2 = lam.lam_I2 * es - (lam.lam_V2 / (p.tau_m * c)) * (em - es)
    lam_V1 = lam.lam_V1 * em
    lam_V2 = lam.lam_V2 * em
    lam_Vout = lam.lam_Vout * eo
    return AdjState(lam_V1, lam_V2, lam_I2, lam_Vout)


def _sample_adjoint_back(lam_end: AdjState, deltas: np.ndarray, p: NetworkParams):
    """Co-state at offsets `deltas` back from the interval's right end."""
    return [propagate_adjoint(lam_end, float(dt), p) for dt in deltas]


def _accumulate_interval(iv, lam_end: AdjState, p: NetworkParams, b: np.ndarray,
                         grads: dict, db: np.ndarray, k_sub: int) -> None:
    """Add this interval's contribution to dL/dtau_m and dL/db (-> dL/dW_in).

    Integrand uses closed-form forward state from the interval start and closed-form
    co-state from the interval end, sampled on a uniform sub-grid (trapezoid).
    """
    L = iv.s_end - iv.s_start
    if L <= 0.0:
        return
    ssub = np.linspace(0.0, L, k_sub + 1)        # offset from interval start
    ds = L / k_sub
    tau_integrand = np.empty(k_sub + 1)
    db_integrand = np.empty((k_sub + 1, p.N1))
    for j, off in enumerate(ssub):
        z = propagate_state(iv.state_start, b, float(off), p)
        lam = propagate_adjoint(lam_end, float(L - off), p)
        vdot1, vdot2 = membrane_vdot(z, b, p)
        tau_integrand[j] = lam.lam_V1 @ vdot1 + lam.lam_V2 @ vdot2
        db_integrand[j] = lam.lam_V1
    w = np.full(k_sub + 1, ds)
    w[0] *= 0.5
    w[-1] *= 0.5
    grads["tau_m"] += -(1.0 / p.tau_m) * float(w @ tau_integrand)
    db += (1.0 / p.tau_m) * (w @ db_integrand)


def _apply_event(ev, lam: AdjState, p: NetworkParams, grads: dict) -> None:
    """Accumulate exact event-sum gradients and apply the transposed spike jump.

    `lam` holds lambda(s_k^+) (post-event side); the jump rewrites only the firing
    neuron's lambda_V to lambda(s_k^-). Event-sum gradients sample the unchanged
    co-state components, so ordering vs the jump is irrelevant for them.
    """
    n = ev.n
    if ev.layer == 2:
        coupling = -float((p.W_out[:, n] / p.tau_out) @ lam.lam_Vout)
        lam_vn = lam.lam_V2[n]
        jn = lam_vn + (p.theta * lam_vn + p.tau_m * coupling) / (p.tau_m * ev.vdot)
        grads["W_out"][:, n] += lam.lam_Vout      # dL/dW_out[p,n] += lambda_Vout,p^+
        grads["theta"] += -jn                     # dL/dtheta = -sum_k J_k
        lam.lam_V2[n] = jn
    else:  # layer 1
        # The presynaptic kick raises postsynaptic I2, which instantaneously jumps
        # dV2/ds = (-V2+I2)/tau_m as well as the I2 decay -> couple back through both
        # lambda_V2 (the W2/tau_m term) and lambda_I2 (the -W2/tau_s term).
        coupling = (float((p.W2[:, n] / p.tau_m) @ lam.lam_V2)
                    - float((p.W2[:, n] / p.tau_s) @ lam.lam_I2))
        lam_vn = lam.lam_V1[n]
        jn = lam_vn + (p.theta * lam_vn + p.tau_m * coupling) / (p.tau_m * ev.vdot)
        grads["W2"][:, n] += lam.lam_I2           # dL/dW2[i,n] += lambda_I2,i^+
        grads["theta"] += -jn
        lam.lam_V1[n] = jn


def backward(p: NetworkParams, u: np.ndarray, target: np.ndarray,
             fwd: ForwardResult, k_sub: int = 64) -> dict:
    """Exact gradient dL/d{W_in, W2, W_out, tau_m, theta} via one backward sweep."""
    grads = {
        "W_in": np.zeros_like(p.W_in),
        "W2": np.zeros_like(p.W2),
        "W_out": np.zeros_like(p.W_out),
        "tau_m": 0.0,
        "theta": 0.0,
    }
    db = np.zeros(p.N1)
    b = fwd.b
    # Terminal co-state = flow-matching residual at the window end.
    lam = AdjState(
        lam_V1=np.zeros(p.N1),
        lam_V2=np.zeros(p.N2),
        lam_I2=np.zeros(p.N2),
        lam_Vout=2.0 * (fwd.Vout_S - target),
    )
    E = len(fwd.events)
    for i in range(E, -1, -1):                    # intervals right-to-left
        iv = fwd.intervals[i]
        _accumulate_interval(iv, lam, p, b, grads, db, k_sub)
        lam = propagate_adjoint(lam, iv.s_end - iv.s_start, p)
        if i >= 1:                                # event i at this interval's left end
            _apply_event(fwd.events[i - 1], lam, p, grads)
    grads["W_in"] = np.outer(db, u)
    return grads
