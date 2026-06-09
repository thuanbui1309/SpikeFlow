"""Wide dense (fully-connected) depth-2 spiking velocity net for generation.

This is the generation-time topology: a wide, DENSE feedforward network with NO
convolution, pooling, or weight sharing. The flow-matching state x_t is flattened
and concatenated with the scalar flow time t to form the direct (x_t, t) input;
two LIF hidden layers feed the non-spiking leaky-integrator readout whose final
membrane potential V_out(S) is the m-dimensional velocity vector.

Construction is a thin wrapper over ``init_params``: setting ``d = m`` makes both
required shapes fall out at once - the input width d_in = d + 1 = m + 1 holds the
(flatten(x_t), t) concat, and the readout W_out has shape (d, n2) = (m, n2), so
the readout produces exactly m velocity components. All dynamical hyper-parameters
(tau_m, tau_s, tau_out, theta, window/S, bias, hidden scales) keep the validated
``init_params`` defaults; only the readout scale is overridden.

The readout scale is set so the implementable coding constant
C_impl = (S/(2 tau_out)) * ||W_out||_1 lands in O(1e2-1e3) rather than the
five-figure value the default readout scale would produce: with S/(2 tau_out)=1
the constant equals ||W_out||_1, whose expectation is scale * (m * n2) * sqrt(2/pi),
so a scale near 0.0117 keeps the readout well below the VAE-fallback regime.
"""

from __future__ import annotations

from .params import NetworkParams, init_params


def init_fc_params(
    seed: int,
    n1: int = 32,
    n2: int = 32,
    m: int = 3072,
    w_out_scale: float = 0.0117,
) -> NetworkParams:
    """Build the wide dense depth-2 spiking velocity net.

    Args:
        seed: RNG seed for weight initialisation.
        n1: layer-1 (current-driven LIF) width.
        n2: layer-2 (synaptic LIF) width.
        m: velocity dimension = flattened-state size. Drives both d_in = m + 1
            (the (x_t, t) concat input) and the readout rows (W_out shape (m, n2)).
        w_out_scale: signed readout-weight scale, sized so the readout coding
            constant stays in O(1e2-1e3).

    Returns the ``NetworkParams`` unchanged from ``init_params`` (d = m); all
    other dynamical constants keep their validated defaults.
    """
    return init_params(seed=seed, d=m, n1=n1, n2=n2, w_out_scale=w_out_scale)
