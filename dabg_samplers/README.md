# Reuse the paper's samplers

This starter runs local MCMC, replica exchange (RE), and fixed-grid sequential
Monte Carlo (SMC) on the paper's `BracketTwoModes` distribution. It runs on CPU
by default. It also exposes a small interface for replacing the target later.

## Start here

Use Python 3.10 or newer. Open a terminal in this `dabg_samplers/` folder, then run:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_sampling
python run_example.py
```

On Windows, activate with `.venv\Scripts\activate` instead.

For a shorter execution check:

```sh
python run_example.py --quick --output quick_results
```

The example saves `summary.json`, `comparison.png`, and `samples.pt` in
`example_results/`. The plot projects the four-dimensional samples onto the
first two coordinates. The quick run verifies execution; its small budgets
are not intended to establish convergence. For NVIDIA CUDA, add `--device cuda`
after installing a compatible PyTorch build.

## The first target

The defaults match the smallest dimension and separation in the paper's sweep:

```
dimension = 4
a = 0.5
target = (2/3) N(-a * ones(4), diag(v))
       + (1/3) N(+a * ones(4), diag(reverse(v)))
v = linspace(0.01, 0.2, 4)
```

`targets.py` implements that same density and its exact gradient without pulling
in the original distribution class's diffusion and evaluation dependencies.
The RE/SMC base is a zero-mean diagonal Gaussian with the target's marginal
variances, as in the original experiment scripts.

The mode diagnostic assigns each sample to the component with the greater
**unweighted** Gaussian density, matching the paper's partition. This region's
probability is close to, but not exactly, the component weight 2/3 when modes
overlap. The example therefore also draws 50,000 direct reference samples and
reports errors against that region-mass estimate, which itself has Monte Carlo
error. At probability 2/3 its IID standard error is about 0.0021.

MALA starts all chains at the heavier mode, following the original experiment's
initialization. RE and SMC initialize from the easy Gaussian base. Local MCMC
can remain stuck in one mode; that is a useful sampling diagnosis. RE and SMC
also need tuning when separation or dimension increases. Comparable sample
counts do not imply comparable compute budgets, and the plots and mode masses
are not a complete convergence assessment.

## Call the samplers yourself

```python
import torch
from targets import BracketTwoModes, Gaussian
from sampling import run_mcmc, run_re, run_smc

torch.manual_seed(42)
target = BracketTwoModes(dim=4, a=0.5)
base = Gaussian(torch.zeros(4), target.dist.variance)

mcmc = run_mcmc(target, target.means[0].repeat(16, 1), kernel="mala")
re = run_re(target, base)
smc = run_smc(target, base)

# MCMC and RE: samples[draw, chain, coordinate].
print(mcmc["samples"].shape)
print(re["samples"].shape)

# SMC: one final particle cloud, samples[particle, coordinate].
print(smc["samples"].shape)
print(smc["weights"].sum())  # normalized to 1
smc_mean = (smc["samples"] * smc["weights"][:, None]).sum(0)
```

Local kernels are `mala`, `rwmh`, and `ula`. MALA and RWMH include Metropolis
corrections. ULA does not, so its stationary law generally has discretization
bias. For MALA/ULA, `step_size=h` means proposal drift `h*gradient` and noise
standard deviation `sqrt(2*h)`. For RWMH, `step_size` is the proposal standard
deviation. RE and SMC use the original MALA kernel for local moves.

The local runner adapts one shared step size during warmup only. RE and SMC
keep their step sizes fixed; inspect acceptance and tune explicitly. RE's
upstream diagnostics give the most recent local and swap acceptance values,
not time-averaged acceptance rates. Every `swap_every`-th RE iteration performs
a swap instead of a local move, so adjacent saved states can repeat.

This starter uses the geometric path
`log pi_t = t*log base + (1-t)*log target`, where `t=0` is the target.
Both endpoints are included exactly. These are the classic RE and SMC methods;
the diffusion samplers and adaptive-temperature SMC are outside this starter.

## Change the target later

Provide a vectorized PyTorch log density returning one scalar per row:

```python
from targets import Target

my_target = Target(lambda x: -0.5 * x.square().sum(dim=-1))
# Optional: Target(log_prob_fn, score_fn=analytical_gradient_fn)

result = run_mcmc(my_target, torch.randn(16, 4))
result_re = run_re(my_target, base)
result_smc = run_smc(my_target, base)
```

An unknown additive normalizing constant is fine. Automatic gradients are
computed with PyTorch, including inside the RE implementation's `no_grad`
blocks. Each log-density row must depend only on the corresponding input row.
These examples assume smooth, finite log densities on all of real space.
Constrained targets require a suitable transformation or boundary treatment.
Use the same dimension, device, and floating-point dtype for the target,
base, and initial positions. RE/SMC require an easy base with `sample` and
`log_prob_and_grad`, whose support covers the target.

## Source and changes

Upstream: https://github.com/h2o64/dabg

Pinned commit: `b273ca39bee5762845cf2948ce28a45ca1354aa3`.
The original MIT license is included in `LICENSE`. The vendored files are adapted from that commit.
The paper is *Improving the evaluation of samplers on
multi-modal targets*, Louis Grenioux, Maxence Noble, and Marylou Gabrié,
Frontiers in Probabilistic Inference at ICLR 2025:
https://arxiv.org/abs/2504.08916.
