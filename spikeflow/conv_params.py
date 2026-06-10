"""Small CONVOLUTIONAL spiking velocity net, materialised as a structured dense net.

The event-driven simulator (`forward.simulate`) runs a fixed two-hidden-layer LIF
skeleton on dense weight matrices W_in [N1, d_in], W2 [N2, N1], W_out [m, N2]. A
convolution is just a *structured* dense map: weight sharing replicates one kernel
across spatial positions (so the same value appears in many matrix entries) and
local connectivity zeroes the rest. Pooling/strided conv aggregates a spatial
window of upstream units into one downstream drive. So we build the conv topology
ONCE as those three matrices and hand them to the SAME simulator and the SAME exact
adjoint -- no change to the event engine, no convolution logic duplicated.

This is what the conv stage isolates relative to the dense FC net: (1) weight
sharing -- spatially symmetric input patches give layer-1 neurons identical drives,
hence simultaneous threshold crossings (ties) the scheduler must order
deterministically; (2) spatial aggregation -- the strided layer-2 conv sums a k2 x k2
window of layer-1 spikes into one layer-2 drive (a learnable strided-conv pooling, the
modern downsample; not max/avg pooling). Everything else (readout snap discipline,
leaky-integrator readout, adjoint) is byte-identical to the FC path.

Topology (encoder -> readout, all weight-shared):
  input (C0,H0,W0) + t  --conv k1,stride1-->  layer1 (C1,H1,W1)   [signed kernel, current-driven LIF]
                        --conv k2,stride2-->  layer2 (C2,H2,W2)   [positive kernel, synaptic LIF; strided -> pooling]
                        --transposeconv  -->  readout (C0,H0,W0)  [signed kernel, leaky integrator -> pixel velocity]
The flatten convention is channel-major row-major: idx(c,h,w) = (c*H + h)*W + w, used
identically for the input image, every hidden map, and the m-dim velocity output.

Fan-in scaling (the lesson from the dense net): each weight block is scaled by
1/sqrt(fan_in) of its receptive field so the per-neuron drive stays O(1) at any
channel/kernel size; without it the spike count explodes and the event sim is
intractable. ALWAYS probe one simulate at full scale before a sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import NetworkParams


@dataclass
class ConvSpec:
    """Spatial shapes of the materialised conv net (for reports / readout C_impl)."""
    img: tuple        # (C0, H0, W0) input image = velocity shape
    l1: tuple         # (C1, H1, W1) layer-1 feature map
    l2: tuple         # (C2, H2, W2) layer-2 feature map
    k1: int
    stride1: int
    k2: int
    stride2: int

    @property
    def N1(self) -> int:
        return int(np.prod(self.l1))

    @property
    def N2(self) -> int:
        return int(np.prod(self.l2))

    @property
    def m(self) -> int:
        return int(np.prod(self.img))


def _idx(c: int, h: int, w: int, H: int, W: int) -> int:
    """Channel-major row-major flatten: (c*H + h)*W + w."""
    return (c * H + h) * W + w


def _conv_unroll(kernel: np.ndarray, in_shape: tuple, stride: int) -> tuple[np.ndarray, tuple]:
    """Dense [N_out, N_in] matrix of a 'valid' strided conv that DOWNSAMPLES.

    kernel: [C_out, C_in, k, k]. Shared across every output spatial position, so the
    same kernel value lands in one matrix entry per (out-position, kernel-tap) pair --
    that replication IS the weight sharing. Returns (M, out_shape).
    """
    C_out, C_in, k, _ = kernel.shape
    Ci, Hi, Wi = in_shape
    assert Ci == C_in, "conv in-channel mismatch"
    H_out = (Hi - k) // stride + 1
    W_out = (Wi - k) // stride + 1
    M = np.zeros((C_out * H_out * W_out, Ci * Hi * Wi))
    for co in range(C_out):
        for ho in range(H_out):
            for wo in range(W_out):
                o = _idx(co, ho, wo, H_out, W_out)
                for ci in range(C_in):
                    for dh in range(k):
                        for dw in range(k):
                            i = _idx(ci, ho * stride + dh, wo * stride + dw, Hi, Wi)
                            M[o, i] = kernel[co, ci, dh, dw]
    return M, (C_out, H_out, W_out)


def _transpose_unroll(kernel: np.ndarray, in_shape: tuple, stride: int) -> tuple[np.ndarray, tuple]:
    """Dense [N_out, N_in] matrix of a transpose conv that UPSAMPLES (non-overlap stride=k).

    kernel: [C_out, C_in, k, k]. Each input spatial position writes its own k x k patch
    into the output (here stride == k so patches tile without overlap, exactly inverting
    the encoder downsample). Returns (M, out_shape). Used for the pixel readout.
    """
    C_out, C_in, k, _ = kernel.shape
    Ci, Hi, Wi = in_shape
    assert Ci == C_in, "transpose in-channel mismatch"
    H_out = Hi * stride if stride == k else (Hi - 1) * stride + k
    W_out = Wi * stride if stride == k else (Wi - 1) * stride + k
    M = np.zeros((C_out * H_out * W_out, Ci * Hi * Wi))
    for ci in range(C_in):
        for hi in range(Hi):
            for wi in range(Wi):
                i = _idx(ci, hi, wi, Hi, Wi)
                for co in range(C_out):
                    for dh in range(k):
                        for dw in range(k):
                            o = _idx(co, hi * stride + dh, wi * stride + dw, H_out, W_out)
                            M[o, i] = kernel[co, ci, dh, dw]
    return M, (C_out, H_out, W_out)


def init_conv_params(
    seed: int,
    img: tuple = (3, 32, 32),
    c1: int = 4, k1: int = 4, stride1: int = 4,
    c2: int = 4, k2: int = 2, stride2: int = 2,
    tau_m: float = 5.0, tau_s: float = 3.0, tau_out: float = 10.0,
    theta: float = 1.0, window: float = 20.0, bias_level: float = 2.0,
    w_out_scale: float = 0.05, t_scale: float = 0.1,
) -> tuple[NetworkParams, ConvSpec]:
    """Build the small weight-shared conv velocity net as a NetworkParams + ConvSpec.

    Returns the params (usable by `forward.simulate` and `ExactEventAdjoint` unchanged)
    and a ConvSpec recording the spatial shapes. Defaults are CIFAR de-risk scale; pass
    a tiny img (e.g. (3,4,4)) for a local smoke. Kernel scales are fan-in normalised so
    the per-neuron drive stays O(1); readout scale is sized so C_impl lands O(1e2-1e3)
    (re-measure post-hoc -- training adjusts W_out).
    """
    rng = np.random.default_rng(seed)
    C0, H0, W0 = img

    # Layer 1: signed conv, fan-in C0*k1*k1 (+1 for the shared t-input).
    fan1 = C0 * k1 * k1
    K1 = (0.25 / np.sqrt(fan1)) * rng.standard_normal((c1, C0, k1, k1))
    W_in_img, l1 = _conv_unroll(K1, img, stride1)                 # [N1, C0*H0*W0]
    N1 = W_in_img.shape[0]
    # Shared per-output-channel t weight: append one column for the flow time input.
    w_t_chan = t_scale * rng.standard_normal(c1)
    C1, H1, W1 = l1
    t_col = np.empty((N1, 1))
    for co in range(C1):
        for ho in range(H1):
            for wo in range(W1):
                t_col[_idx(co, ho, wo, H1, W1), 0] = w_t_chan[co]
    W_in = np.concatenate([W_in_img, t_col], axis=1)             # [N1, C0*H0*W0 + 1]
    bias = bias_level + 0.1 * rng.standard_normal(N1)

    # Layer 2: positive (excitatory) strided conv = learnable spatial pooling over a
    # k2 x k2 window, fan-in C1*k2*k2. Positive so the synaptic current accumulates and
    # layer 2 actually fires (vacuity guard).
    fan2 = C1 * k2 * k2
    K2 = (0.6 / np.sqrt(fan2)) * np.abs(rng.standard_normal((c2, C1, k2, k2)))
    W2, l2 = _conv_unroll(K2, l1, stride2)                       # [N2, N1]
    C2, H2, W2_ = l2

    # Readout: signed transpose conv, layer-2 map -> pixel velocity (non-overlap tiling).
    up = stride1 * stride2
    Kout = w_out_scale * rng.standard_normal((C0, C2, up, up))
    W_out, out_shape = _transpose_unroll(Kout, l2, up)          # [m, N2]
    assert out_shape == img, f"readout shape {out_shape} != image {img}"

    p = NetworkParams(
        W_in=W_in, bias=bias, W2=W2, W_out=W_out,
        tau_m=float(tau_m), theta=float(theta),
        tau_s=float(tau_s), tau_out=float(tau_out), S=float(window),
    )
    spec = ConvSpec(img=img, l1=l1, l2=l2, k1=k1, stride1=stride1, k2=k2, stride2=stride2)
    return p, spec
