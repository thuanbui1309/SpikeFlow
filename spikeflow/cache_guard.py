"""Cross-step no-cache tripwire for the PF-ODE sampler.

INVARIANT this guard protects:
    Every PF-ODE integration step re-runs the FULL-PRECISION hidden dynamics
    from scratch (a fresh ``simulate`` call inside ``velocity``). The hidden
    (layer-1/layer-2) spike counts and order are produced by the event-driven
    solver and are exact; only the readout-feeding spike TIMES are snapped to a
    1/T grid afterwards. If any caller were to cache a ``ForwardResult`` (or the
    ``State`` it carries) and reuse it across a step instead of re-simulating,
    that stale, already-time-snapped result would be re-injected into a later
    step. The hidden-time snap would then be applied on top of itself and the
    exact spike-count semantics the sampler depends on would silently break.

How it works (zero risk to the verified sampler):
    ``register`` stores a weak reference to each ForwardResult produced by a
    ``velocity`` call. ``assert_no_survivors`` checks, at every step boundary,
    that none of those weakly-held results are still alive. A ForwardResult is
    acyclic, so under CPython reference counting it is reclaimed the instant the
    local ``fwd`` in ``velocity`` goes out of scope (``velocity`` returns only
    the velocity array, never ``fwd``). For the correct sampler the live set is
    therefore always empty at the boundary -> the assertion is an instant pass
    and can NEVER false-positive. It raises only when some caller holds a strong
    reference to a result across a step, i.e. exactly the forbidden caching.

Pure stdlib weakref; no gc.collect, no torch.
"""

from __future__ import annotations

import weakref

# Weak references to the ForwardResult objects produced since the last step
# boundary. Weak so registration alone never keeps a result alive.
_live_results: "list[weakref.ref]" = []


def register(result) -> None:
    """Track ``result`` weakly so the next boundary can confirm it was freed."""
    _live_results.append(weakref.ref(result))


def assert_no_survivors() -> None:
    """Raise if any tracked ForwardResult survived to this step boundary.

    A survivor means a strong reference to a per-step ForwardResult is being
    held across a PF-ODE step, which would re-inject already-snapped hidden
    dynamics. Dead references are pruned so the next step starts clean.
    """
    survivors = [ref for ref in _live_results if ref() is not None]
    _live_results.clear()
    if survivors:
        raise RuntimeError(
            f"{len(survivors)} ForwardResult object(s) survived across a PF-ODE "
            "step; a per-step forward result is being cached/held across steps, "
            "which re-injects already-time-snapped hidden dynamics and breaks "
            "exact spike-count semantics. Each step must re-simulate from scratch."
        )
