"""Network parameters and configuration for the spiking velocity field.

The velocity field v_theta(x_t, t) is realised by a small spike-driven network:
two hidden LIF layers feeding a non-spiking leaky-integrator readout whose final
membrane potential is the velocity vector. Trainable parameters validated by the
finite-difference gate are the weights, the shared membrane time constant tau_m,
and the shared firing threshold theta. Synaptic and readout time constants are
fixed hyper-parameters here (the gate targets {W, tau_m, theta}).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class NetworkParams:
    """Trainable arrays/scalars plus fixed dynamical constants.

    Layer 1 is current-driven: each neuron integrates a constant input current
    b = W_in @ u + bias. Layer 2 is synaptic: layer-1 spikes kick the synaptic
    current I (I += W2 column). The readout integrates layer-2 spikes into V_out
    (V_out += W_out column) and never spikes; v_theta = V_out(S).
    """

    # Trainable
    W_in: np.ndarray   # [N1, d_in]   d_in = d + 1  (input is (x_t, t))
    bias: np.ndarray   # [N1]         fixed-by-default offset keeping layer 1 active
    W2: np.ndarray     # [N2, N1]
    W_out: np.ndarray  # [m, N2]      m = d  (velocity dimension)
    tau_m: float       # shared membrane time constant (hidden layers)
    theta: float       # shared spike threshold

    # Fixed dynamical constants
    tau_s: float       # synaptic current time constant (layer 2)
    tau_out: float     # readout leak time constant  (MUST be finite: the leak is
                       # what makes V_out(S) smooth in spike times, not a flat count)
    S: float           # internal-time window length
    v_rest: float = 0.0

    @property
    def N1(self) -> int:
        return self.W_in.shape[0]

    @property
    def N2(self) -> int:
        return self.W2.shape[0]

    @property
    def m(self) -> int:
        return self.W_out.shape[0]

    @property
    def d_in(self) -> int:
        return self.W_in.shape[1]

    def copy(self) -> "NetworkParams":
        return replace(
            self,
            W_in=self.W_in.copy(),
            bias=self.bias.copy(),
            W2=self.W2.copy(),
            W_out=self.W_out.copy(),
        )


def init_params(
    seed: int,
    d: int = 2,
    n1: int = 8,
    n2: int = 8,
    tau_m: float = 5.0,
    tau_s: float = 3.0,
    tau_out: float = 10.0,
    theta: float = 1.0,
    window: float = 20.0,
    bias_level: float = 2.0,
    w_in_scale: float = 0.25,
    w2_scale: float = 0.6,
    w_out_scale: float = 0.8,
) -> NetworkParams:
    """Initialise a small but healthily-spiking network.

    Defaults are chosen so layer 1 fires several times over the window (b > theta
    with a comfortable margin -> transversal crossings, V_dot well away from zero)
    and the accumulated drive pushes layer 2 over threshold too, so every gradient
    path (input weights, synaptic weights, readout weights, tau_m, theta) is
    actually exercised. A silent network would make the gate pass vacuously.
    """
    rng = np.random.default_rng(seed)
    d_in = d + 1
    W_in = w_in_scale * rng.standard_normal((n1, d_in))
    bias = bias_level + 0.1 * rng.standard_normal(n1)
    W2 = w2_scale * np.abs(rng.standard_normal((n2, n1)))  # excitatory -> drives layer 2 up
    W_out = w_out_scale * rng.standard_normal((d, n2))      # signed readout weights
    return NetworkParams(
        W_in=W_in,
        bias=bias,
        W2=W2,
        W_out=W_out,
        tau_m=float(tau_m),
        theta=float(theta),
        tau_s=float(tau_s),
        tau_out=float(tau_out),
        S=float(window),
    )
