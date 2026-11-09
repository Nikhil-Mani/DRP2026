"""Reusable entry points around the original local MCMC, RE and fixed-grid SMC."""

import math
import torch
from dabg.mcmc.samplers import mala_step, rwmh_step, ula_step
from dabg.mcmc.utils import heuristics_step_size
from dabg.re.classic import re_sampler
from dabg.smc.classic import smc_sampler
from dabg.smc.utils import getter_log_prob_and_grads_tempering


def _count(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _step(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("step_size must be finite and positive.")


def _positions(x):
    if x.ndim != 2 or min(x.shape) < 1 or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("Initial positions must be a finite floating tensor (n_chains, dim).")
    return x.detach().clone()


def run_mcmc(target, x_init, *, kernel="mala", n_samples=1000, warmup=500,
             step_size=0.005, thin=1, adapt_warmup=True):
    """Run independent local chains. Return samples as (draw, chain, dim).

    MALA/ULA: step_size h means proposal x + h*score + sqrt(2h)*noise.
    RWMH: step_size is the proposal standard deviation, not its variance.
    Step-size adaptation is restricted to warmup and then frozen.
    ULA is discretization biased and has no Metropolis correction.
    """
    _count("n_samples", n_samples)
    _count("warmup", warmup, 0)
    _count("thin", thin)
    _step(step_size)
    if kernel not in ("mala", "rwmh", "ula"):
        raise ValueError("kernel must be 'mala', 'rwmh' or 'ula'.")
    x = _positions(x_init)
    steps = torch.full((x.shape[0], 1), step_size, device=x.device, dtype=x.dtype)
    if kernel == "rwmh":
        log_prob = target.log_prob(x).detach()
    else:
        log_prob, gradient = target.log_prob_and_grad(x)
    draws = []
    acceptance_sum = torch.zeros((), device=x.device, dtype=x.dtype)
    for i in range(warmup + n_samples * thin):
        if kernel == "rwmh":
            x, log_prob, log_acc = rwmh_step(x, log_prob, target.log_prob, steps)
        elif kernel == "mala":
            x, log_prob, gradient, log_acc = mala_step(
                x, log_prob, gradient, target.log_prob_and_grad, steps)
        else:
            x, log_prob, gradient = ula_step(
                x, log_prob, gradient, target.log_prob_and_grad, steps)
            log_acc = None
        if log_acc is not None:
            acc = log_acc.clamp(max=0).exp().mean()
            if i < warmup and adapt_warmup:
                goal = 0.75 if kernel == "mala" else 0.234
                # A shared step size avoids state-dependent steps during sampling.
                steps = heuristics_step_size(steps, acc.log().reshape(1), target_acceptance=goal)
            elif i >= warmup:
                acceptance_sum += acc
        if i >= warmup and (i - warmup + 1) % thin == 0:
            draws.append(x.detach().clone())
    samples = torch.stack(draws)
    if not torch.isfinite(samples).all():
        raise RuntimeError("Non-finite MCMC samples; check the target and reduce step_size.")
    return {"samples": samples, "weights": None, "diagnostics": {
        "kernel": kernel, "step_size": steps[0, 0].item(),
        "mean_acceptance_probability": None if kernel == "ula" else
            (acceptance_sum / (n_samples * thin)).item()}}


def _times(n_levels, n_chains, device, dtype):
    # Upstream convention: t=0 is the target; t=1 is the easy base.
    return torch.linspace(0, 1, n_levels, device=device, dtype=dtype).view(-1, 1, 1).expand(-1, n_chains, 1)


def run_re(target, base, *, n_chains=16, n_samples=2000, warmup=500,
           n_levels=32, swap_every=8, step_size=0.005):
    """Geometric-path replica exchange with original MALA and swap kernels.

    Returns target-replica samples only, shaped (draw, chain, dim).
    Step sizes stay fixed throughout; tune step_size for your target.
    The original loop replaces every swap_every-th local step with swaps.
    """
    for name, value, minimum in [("n_chains", n_chains, 1), ("n_samples", n_samples, 1),
                                 ("warmup", warmup, 0), ("n_levels", n_levels, 3),
                                 ("swap_every", swap_every, 2)]:
        _count(name, value, minimum)
    if warmup + n_samples <= swap_every:
        raise ValueError("RE needs more total steps than swap_every to exercise both pairings.")
    _step(step_size)
    x = _positions(base.sample((n_chains,)))
    times = _times(n_levels, n_chains, x.device, x.dtype)
    samples, _, _, diagnostics = re_sampler(
        x_init=x, times=times,
        log_prob_and_grads=getter_log_prob_and_grads_tempering(base, target),
        swap_frequency=swap_every, n_warmup_mcmc_steps=warmup,
        n_mcmc_steps=n_samples, step_sizes_per_noise=torch.full_like(times, step_size),
        target_acceptance=0.0, verbose=False)
    # Original log_Z estimates follow its time ordering; not exposed by this wrapper.
    if not torch.isfinite(samples).all():
        raise RuntimeError("RE returned non-finite samples.")
    return {"samples": samples, "weights": None, "diagnostics": diagnostics}


def run_smc(target, base, *, n_particles=1024, n_levels=64,
            mcmc_steps=16, step_size=0.005, resample_threshold=0.5):
    """Fixed-grid annealed SMC with original MALA and systematic resampling.

    Returns the final particle cloud (particle, dim) and its normalized weights.
    Use these weights when estimating expectations. Step sizes are fixed.
    """
    for name, value, minimum in [("n_particles", n_particles, 2),
                                 ("n_levels", n_levels, 2), ("mcmc_steps", mcmc_steps, 1)]:
        _count(name, value, minimum)
    _step(step_size)
    if not 0 <= resample_threshold <= 1:
        raise ValueError("resample_threshold must lie in [0, 1].")
    x = _positions(base.sample((n_particles,)))
    times = _times(n_levels, n_particles, x.device, x.dtype)
    samples, weights, _, log_z, diagnostics = smc_sampler(
        x_init=x, times=times,
        log_prob_and_grads=getter_log_prob_and_grads_tempering(base, target),
        n_warmup_mcmc_steps=0, n_mcmc_steps=mcmc_steps,
        step_sizes_per_noise=torch.full((n_particles, 1), step_size, device=x.device, dtype=x.dtype),
        reweight_threshold=resample_threshold, target_acceptance=0.0, verbose=False)
    final_samples = samples[-1]
    if not torch.isfinite(final_samples).all() or not torch.isfinite(weights).all():
        raise RuntimeError("SMC returned non-finite values.")
    diagnostics["final_weight_ess"] = (1 / weights.square().sum()).item()
    diagnostics["log_z_ratio_estimate"] = log_z.item()
    return {"samples": final_samples, "weights": weights, "diagnostics": diagnostics}
