"""Net-agnostic stability/operational measures for the spiking velocity field.

Two scalar diagnostics shared by every net topology (dense FC here, conv later):

* ``c_impl_suffix_sum`` - the implementable readout coding constant. The smooth
  leaky-integrator readout V_out(S) = sum_k W_out[:,n_k] exp(-(S - s_k)/tau_out)
  contracts each spike-time perturbation by the leak factor, so its sensitivity
  to a 1/T time snap is bounded by (S/(2 tau_out)) * ||W_out||_1. Only the
  readout snap survives (the hidden dynamics are re-run exactly each step), so
  this is the full L1 norm of the readout weights times the half-grid leak
  prefactor - no per-layer product term.

* ``mu_min`` - the smallest transversality margin among readout-feeding events.
  Each readout-feeding (layer-2) crossing has a membrane slope vdot at the
  crossing; |vdot| is the denominator that governs whether a small parameter or
  time perturbation can change which side of the threshold the crossing lands
  on. The minimum |vdot| over those events is the worst-case snap-survival
  margin. vdot is recorded by the forward simulator; this never recomputes it.
"""

from __future__ import annotations

import numpy as np

from .forward import ForwardResult
from .params import NetworkParams


def c_impl_suffix_sum(p: NetworkParams) -> float:
    """Readout coding constant (S/(2 tau_out)) * ||W_out||_1."""
    return float((p.S / (2.0 * p.tau_out)) * np.abs(p.W_out).sum())


def mu_min(fwd: ForwardResult) -> float:
    """Smallest |vdot| over readout-feeding (layer-2) events; inf if none."""
    margins = [abs(e.vdot) for e in fwd.events if e.layer == 2]
    if not margins:
        return float("inf")
    return float(min(margins))
