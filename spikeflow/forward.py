"""Event-driven forward simulation of the spiking velocity field.

Between spikes the dynamics are linear constant-coefficient and propagated in
closed form (zero integration error, so a finite-difference of the loss is the
exact continuous-time gradient). Spike times are found exactly: layer 1 has a
single-exponential membrane with a closed-form threshold crossing; layer 2 has a
dual-exponential membrane whose crossing is bracketed and bisected to machine
precision. The simulator records an ordered event log and per-interval start
states so the backward adjoint can replay the trajectory exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .params import NetworkParams


@dataclass
class State:
    V1: np.ndarray
    V2: np.ndarray
    I2: np.ndarray
    Vout: np.ndarray

    def copy(self) -> "State":
        return State(self.V1.copy(), self.V2.copy(), self.I2.copy(), self.Vout.copy())


@dataclass
class Event:
    time: float
    layer: int      # 1 or 2
    n: int          # firing neuron index within its layer
    vdot: float     # membrane slope just before crossing (transversality denominator)


@dataclass
class Interval:
    s_start: float
    s_end: float
    state_start: State


@dataclass
class ForwardResult:
    Vout_S: np.ndarray
    b: np.ndarray
    events: list = field(default_factory=list)
    intervals: list = field(default_factory=list)

    def loss(self, target: np.ndarray) -> float:
        r = self.Vout_S - target
        return float(r @ r)


def zero_state(p: NetworkParams) -> State:
    return State(np.zeros(p.N1), np.zeros(p.N2), np.zeros(p.N2), np.zeros(p.m))


def propagate_state(st: State, b: np.ndarray, dt: float, p: NetworkParams) -> State:
    """Advance the closed-form dynamics by dt assuming no event in (s, s+dt)."""
    if dt <= 0.0:
        return st.copy()
    em = np.exp(-dt / p.tau_m)
    es = np.exp(-dt / p.tau_s)
    eo = np.exp(-dt / p.tau_out)
    V1 = b + (st.V1 - b) * em
    A = st.I2 * p.tau_s / (p.tau_s - p.tau_m)
    V2 = (st.V2 - A) * em + A * es
    I2 = st.I2 * es
    Vout = st.Vout * eo
    return State(V1, V2, I2, Vout)


def membrane_vdot(st: State, b: np.ndarray, p: NetworkParams):
    """Membrane time-derivatives of hidden neurons at the given state."""
    vdot1 = (-st.V1 + b) / p.tau_m
    vdot2 = (-st.V2 + st.I2) / p.tau_m
    return vdot1, vdot2


def _layer1_next_cross(V1: np.ndarray, b: np.ndarray, p: NetworkParams) -> tuple[float, int]:
    """Earliest closed-form upward crossing of theta among layer-1 neurons."""
    best_dt, best_i = np.inf, -1
    for i in range(len(V1)):
        if b[i] <= p.theta or V1[i] >= p.theta:
            continue
        # V(s)=b+(V0-b)e^{-s/tau_m}=theta  ->  s = -tau_m ln((theta-b)/(V0-b))
        dt = -p.tau_m * np.log((p.theta - b[i]) / (V1[i] - b[i]))
        if 0.0 < dt < best_dt:
            best_dt, best_i = dt, i
    return best_dt, best_i


def _v2_at(V0: float, I0: float, s: float, p: NetworkParams) -> float:
    em = np.exp(-s / p.tau_m)
    es = np.exp(-s / p.tau_s)
    A = I0 * p.tau_s / (p.tau_s - p.tau_m)
    return (V0 - A) * em + A * es


def _layer2_next_cross(V2: np.ndarray, I2: np.ndarray, horizon: float,
                       p: NetworkParams, grid_dt: float) -> tuple[float, int, float]:
    """Earliest upward crossing of theta among layer-2 neurons within (0, horizon].

    Detection: evaluate V on a fine grid (vectorised over neurons), pick the earliest
    first upward-crossing cell, then bisect to ~1e-13 so the spike time is a smooth
    function of parameters.
    """
    if horizon <= 0.0:
        return np.inf, -1, 0.0
    ngrid = max(2, int(np.ceil(horizon / grid_dt)) + 1)
    grid = np.linspace(0.0, horizon, ngrid)                 # [G]
    A = I2 * p.tau_s / (p.tau_s - p.tau_m)                  # [N2]
    em = np.exp(-grid[:, None] / p.tau_m)                   # [G,1]
    es = np.exp(-grid[:, None] / p.tau_s)
    Vg = (V2 - A)[None, :] * em + A[None, :] * es           # [G, N2]
    below = Vg[:-1, :] < p.theta
    above = Vg[1:, :] >= p.theta
    cross = below & above                                   # first-cell crossings
    best_dt, best_n, best_vdot = np.inf, -1, 0.0
    for n in range(V2.shape[0]):
        cells = np.nonzero(cross[:, n])[0]
        if cells.size == 0:
            continue
        g = cells[0]
        lo, hi = grid[g], grid[g + 1]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _v2_at(V2[n], I2[n], mid, p) < p.theta:
                lo = mid
            else:
                hi = mid
        s_cross = 0.5 * (lo + hi)
        if s_cross < best_dt:
            I_cross = I2[n] * np.exp(-s_cross / p.tau_s)
            vdot = (-p.theta + I_cross) / p.tau_m
            if vdot > 0.0:
                best_dt, best_n, best_vdot = s_cross, n, vdot
    return best_dt, best_n, best_vdot


def simulate(p: NetworkParams, u: np.ndarray, grid_dt: float = 0.05) -> ForwardResult:
    """Run the spiking velocity field on input u=(x_t, t); return v_theta=V_out(S)."""
    b = p.W_in @ u + p.bias
    st = zero_state(p)
    res = ForwardResult(Vout_S=None, b=b)
    s_now = 0.0
    eps = 1e-12
    while True:
        dt1, i1 = _layer1_next_cross(st.V1, b, p)
        # Between layer-1 spikes the layer-2 drive (I2) is fixed, so any layer-2
        # crossing before the next layer-1 spike is final; cap the search there.
        horizon = min(p.S - s_now, dt1)
        dt2, n2, vdot2 = _layer2_next_cross(st.V2, st.I2, horizon, p, grid_dt)
        t1 = s_now + dt1
        t2 = s_now + dt2
        next_t = min(t1, t2, p.S)
        dt = next_t - s_now
        res.intervals.append(Interval(s_now, next_t, st.copy()))
        st = propagate_state(st, b, dt, p)
        if next_t >= p.S - eps:
            break
        if t1 <= t2:  # layer-1 spike
            vdot1 = (b[i1] - p.theta) / p.tau_m
            res.events.append(Event(next_t, 1, i1, vdot1))
            st.V1[i1] = p.v_rest
            st.I2 = st.I2 + p.W2[:, i1]
        else:         # layer-2 spike
            res.events.append(Event(next_t, 2, n2, vdot2))
            st.V2[n2] = p.v_rest
            st.Vout = st.Vout + p.W_out[:, n2]
        s_now = next_t
    res.Vout_S = st.Vout
    return res
