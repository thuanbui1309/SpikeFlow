"""nn.Module front-end for the spiking velocity field with the exact adjoint.

Exposes the trainable set {W_in, W2, W_out} as nn.Parameter and keeps the rest
frozen (bias, tau_m, theta as buffers; tau_s/tau_out/S/v_rest as a plain config
dict). This mirrors exactly what train.py optimises -- the adjoint and the gate
certify gradients for {W_in, W2, W_out, tau_m, theta}, but in the training loop
tau_m/theta are held fixed, so here they are buffers (state, not parameters).

torch is imported only in this module and torch_core (never in __init__), so
`import spikeflow` works without torch installed.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .params import NetworkParams
from .torch_core import ExactEventAdjoint


class SpikeVelocityModule(nn.Module):
    """Wrap a NetworkParams as an nn.Module; velocity_exact() runs the exact adjoint."""

    def __init__(self, p: NetworkParams):
        super().__init__()
        # Trainable weights (the set train.py optimises).
        self.W_in = nn.Parameter(torch.tensor(p.W_in, dtype=torch.float64))
        self.W2 = nn.Parameter(torch.tensor(p.W2, dtype=torch.float64))
        self.W_out = nn.Parameter(torch.tensor(p.W_out, dtype=torch.float64))

        # FROZEN: bias gets no adjoint gradient; tau_m/theta are gate-differentiable
        # but held fixed during training -> buffers (saved state, not optimised).
        self.register_buffer("bias", torch.tensor(p.bias, dtype=torch.float64))
        self.register_buffer("tau_m", torch.tensor(float(p.tau_m), dtype=torch.float64))
        self.register_buffer("theta", torch.tensor(float(p.theta), dtype=torch.float64))

        # Fixed scalar config never differentiated by the adjoint.
        self.fixed = {
            "tau_s": float(p.tau_s),
            "tau_out": float(p.tau_out),
            "S": float(p.S),
            "v_rest": float(p.v_rest),
        }

    def velocity_exact(self, u, target=None):
        """v_theta(u) = V_out(S) via the event-driven forward + exact-adjoint backward."""
        return ExactEventAdjoint.apply(
            self.W_in, self.bias, self.W2, self.W_out,
            self.tau_m, self.theta, u, target, self.fixed,
        )
