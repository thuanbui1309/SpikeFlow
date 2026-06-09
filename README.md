# SpikeFlow

Exact event-driven adjoint for a **spiking velocity field** trained by flow matching, in pure
NumPy. Two self-contained experiments:

1. **Finite-difference gate** — certifies the hand-derived continuous-time adjoint (the gradient of
   the simulation-free flow-matching loss w.r.t. `{W, tau_m, theta}`) against central finite
   differences and an independent forward-mode sensitivity. No autodiff, no surrogate gradient.
2. **Generation bound (W2 vs resolution T)** — trains the spiking velocity field and measures the
   Wasserstein-2 distance of the spike-resolution sampler to the target as a function of the
   temporal-resolution (spike-time) quantization `T`, expected to follow `a + b/T`.

## Model

A small spike-driven network realises `v_theta(x_t, t)`:

- **Layer 1** — current-driven LIF, input `b = W_in @ (x_t, t) + bias`, closed-form threshold crossing.
- **Layer 2** — synaptic LIF `(V, I)`; layer-1 spikes kick `I`; dual-exponential crossing (bisected).
- **Readout** — non-spiking **leaky integrator**; `v_theta = V_out(S)`. The finite leak is mandatory:
  a pure integrator of spike kicks is a flat spike *count*, whose gradient in spike times vanishes.

The forward simulation is event-driven with exact inter-spike propagation; the backward pass is a
single closed-form sweep with the transposed saltation jump at each spike.

## Setup (uv)

```bash
uv sync                 # creates .venv from pyproject.toml (numpy, scipy)
uv sync --extra plot    # also install matplotlib (figure plotting)
```

## Run

```bash
# 1. Gradient gate (certification). Fast; pure NumPy.
uv run python run_finite_diff_gate.py

# 1b. Torch-path gate: validates the torch.autograd wrapper against the NumPy
#     finite-difference oracle, float64 end-to-end (needs torch installed).
uv run python run_torch_gate.py

# 2. Generation bound (headline W2-vs-T). Scale n-samp / n-workers on a many-core server.
uv run python run_toy_w2.py --quick                              # local smoke test
uv run python run_toy_w2.py --n-workers 32 --n-samp 4000 --save results/headline.npz

# 3. Parallel configuration search (find the regime with the smallest W2 floor).
uv run python run_config_sweep.py --n-workers 8
uv run python run_config_sweep.py --quick                        # local smoke test

# 4. Render the generation-bound figure from a saved run (needs the 'plot' extra).
uv sync --extra plot
uv run python plot_headline.py results/headline.npz --out results/headline
```

For fast iteration use a smaller, looser eval (`--n-steps 8 --n-samp 2500 --n-rep 6`);
for the final figure raise `--n-samp` / `--n-rep` so the per-T standard error shrinks.

`--n-workers` parallelises the population sampler (run_toy_w2) or the configurations
(run_config_sweep) across CPU cores — set it to the server core count.

## Layout

```
spikeflow/
  params.py             network config + initialisation
  forward.py            event-driven LIF forward simulation (exact spike times)
  adjoint.py            exact backward adjoint (the contribution)
  forward_sensitivity.py  independent forward-mode oracle (gate cross-check)
  finite_diff.py        central-difference oracle
  sampling.py           flow-matching training-pair sampler
  generation.py         spike-resolution velocity, PF-ODE sampler, 1-D W2 (incl. analytic-target)
  theorem2.py           parallel W2-vs-T pipeline + a + b/T fit
  gate_metrics.py       relative-error + grouping helpers
run_finite_diff_gate.py gate driver
run_toy_w2.py           generation-bound driver (parts A / B' / B)
run_config_sweep.py     parallel regime search
plot_headline.py        render the 3-panel generation-bound figure from a saved .npz
```
