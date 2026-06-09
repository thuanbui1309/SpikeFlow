"""Forward-mode (tangent) sensitivity oracle for a single parameter direction.

Independent computation of dL/dtheta by propagating the state tangent w = dz/dtheta
forward through the trajectory: the variational ODE on each inter-spike interval
plus the saltation jump at each spike (the same implicit-function spike-time shift
the adjoint uses, here applied in the forward direction). Cross-checks the backward
adjoint and matches the cost-vs-direction trade-off discussed for deep velocity
networks (forward-mode is O(#params) per direction; the adjoint is one sweep).

Used to localise discrepancies against finite differences. f is linear in the
state within an interval, so the Jacobian is constant; the tangent is integrated
with fine RK4 on the closed-form state trajectory.
"""

from __future__ import annotations

import numpy as np

from .forward import ForwardResult, propagate_state
from .params import NetworkParams


class Tangent:
    __slots__ = ("V1", "V2", "I2", "Vout")

    def __init__(self, p: NetworkParams):
        self.V1 = np.zeros(p.N1)
        self.V2 = np.zeros(p.N2)
        self.I2 = np.zeros(p.N2)
        self.Vout = np.zeros(p.m)


def _f_dot(w_V1, w_V2, w_I2, w_Vout, p):
    """Jacobian action J @ w (f is linear in state, so J is state-independent)."""
    d_V1 = -w_V1 / p.tau_m
    d_V2 = (-w_V2 + w_I2) / p.tau_m
    d_I2 = -w_I2 / p.tau_s
    d_Vout = -w_Vout / p.tau_out
    return d_V1, d_V2, d_I2, d_Vout


def _source_tau_m(z, b, p):
    """df/dtau_m at state z (nonzero only in membrane components)."""
    g_V1 = -(-z.V1 + b) / p.tau_m**2
    g_V2 = -(-z.V2 + z.I2) / p.tau_m**2
    return g_V1, g_V2


def _propagate_tangent(w, iv, b, p, pspec, k_sub):
    """RK4-integrate the tangent across one inter-spike interval."""
    L = iv.s_end - iv.s_start
    if L <= 0.0:
        return
    h = L / k_sub
    kind = pspec[0]
    for step in range(k_sub):
        s0 = step * h

        def deriv(off, wv):
            wV1, wV2, wI2, wVout = wv
            dV1, dV2, dI2, dVout = _f_dot(wV1, wV2, wI2, wVout, p)
            if kind == "tau_m":
                z = propagate_state(iv.state_start, b, off, p)
                gV1, gV2 = _source_tau_m(z, b, p)
                dV1 = dV1 + gV1
                dV2 = dV2 + gV2
            elif kind == "b":
                i = pspec[1]
                dV1 = dV1.copy()
                dV1[i] += 1.0 / p.tau_m
            return (dV1, dV2, dI2, dVout)

        def add(wv, dv, a):
            return tuple(x + a * d for x, d in zip(wv, dv))

        wv = (w.V1, w.V2, w.I2, w.Vout)
        k1 = deriv(s0, wv)
        k2 = deriv(s0 + 0.5 * h, add(wv, k1, 0.5 * h))
        k3 = deriv(s0 + 0.5 * h, add(wv, k2, 0.5 * h))
        k4 = deriv(s0 + h, add(wv, k3, h))
        w.V1 = w.V1 + (h / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        w.V2 = w.V2 + (h / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        w.I2 = w.I2 + (h / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        w.Vout = w.Vout + (h / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])


def _bracket_vector(ev, z_pre, b, p):
    """f^+ - (dg/dz) f^- at the event (nonzero only at firing V and downstream)."""
    bv_V1 = np.zeros(p.N1)
    bv_V2 = np.zeros(p.N2)
    bv_I2 = np.zeros(p.N2)
    bv_Vout = np.zeros(p.m)
    n = ev.n
    if ev.layer == 1:
        bv_V1[n] = b[n] / p.tau_m                 # I_n^-/tau_m with drive=b
        bv_I2 = -p.W2[:, n] / p.tau_s             # downstream current kick
        bv_V2 = p.W2[:, n] / p.tau_m              # kick raises I2 -> jumps dV2/ds = (-V2+I2)/tau_m
    else:
        bv_V2[n] = z_pre.I2[n] / p.tau_m          # I_n^-/tau_m
        bv_Vout = -p.W_out[:, n] / p.tau_out      # downstream readout kick
    return bv_V1, bv_V2, bv_I2, bv_Vout


def _apply_saltation(w, ev, z_pre, b, p, pspec):
    """Forward saltation: w^+ = w^- - e_Vn w^-_Vn + bracket * (w^-_Vn / Vdot) + explicit."""
    n = ev.n
    vdot = ev.vdot
    bvV1, bvV2, bvI2, bvVout = _bracket_vector(ev, z_pre, b, p)
    w_vn = w.V1[n] if ev.layer == 1 else w.V2[n]
    # homogeneous saltation
    if ev.layer == 1:
        w.V1[n] = 0.0
    else:
        w.V2[n] = 0.0
    coef = w_vn / vdot
    w.V1 = w.V1 + coef * bvV1
    w.V2 = w.V2 + coef * bvV2
    w.I2 = w.I2 + coef * bvI2
    w.Vout = w.Vout + coef * bvVout
    # explicit parameter terms at the event
    kind = pspec[0]
    if kind == "theta":
        c = -1.0 / vdot                            # dPhi/dtheta = -1
        w.V1 = w.V1 + c * bvV1
        w.V2 = w.V2 + c * bvV2
        w.I2 = w.I2 + c * bvI2
        w.Vout = w.Vout + c * bvVout
    elif kind == "W2" and ev.layer == 1 and n == pspec[2]:
        w.I2[pspec[1]] += 1.0                       # dg/dW2[i,j], j=n
    elif kind == "W_out" and ev.layer == 2 and n == pspec[2]:
        w.Vout[pspec[1]] += 1.0                     # dg/dW_out[p,n], n=pspec[2]


def forward_sensitivity(p: NetworkParams, u: np.ndarray, target: np.ndarray,
                        fwd: ForwardResult, pspec: tuple, k_sub: int = 64) -> float:
    """dL/dtheta for a single parameter, pspec one of:
    ('tau_m',), ('theta',), ('b', i), ('W2', i, j), ('W_out', p, n)."""
    b = fwd.b
    w = Tangent(p)
    E = len(fwd.events)
    for i in range(E + 1):
        iv = fwd.intervals[i]
        _propagate_tangent(w, iv, b, p, pspec, k_sub)
        if i < E:
            ev = fwd.events[i]
            z_pre = propagate_state(iv.state_start, b, iv.s_end - iv.s_start, p)
            _apply_saltation(w, ev, z_pre, b, p, pspec)
    return float(2.0 * (fwd.Vout_S - target) @ w.Vout)
