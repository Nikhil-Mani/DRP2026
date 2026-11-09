# Classic Sequential Monte Carlo implementation

# Libraries
import math
import torch
from tqdm import trange
from .utils import resampling_idx_systematic
from ..mcmc.utils import heuristics_step_size
from ..mcmc.samplers import mala_step


def smc_sampler(x_init, times, log_prob_and_grads, n_warmup_mcmc_steps, n_mcmc_steps, step_sizes_per_noise,
                reweight_threshold=1.0, target_acceptance=0.75, integrator_type='ei', return_intermediates=False,
                samples_device=None, verbose=True):
    """Sequential Monte Carlo sampler.

    Args:
        * x_init (torch.Tensor of shape (n_particles, *data_shape)): Initial samples
        * times (torch.Tensor of shape (n_levels, n_particles, *data_shape_ones)): Value of the intermediate noise levels
        * log_prob_and_grads (function taking t of shape (n_particles, *data_shape_ones) and x of shape (n_particles, *data_shape)): Intermediate distributions
        * n_warmup_mcmc_steps (int): Number of initial warmup steps for each noise level
        * n_mcmc_steps (int): Number of steps for each noise level
        * step_sizes_per_noise (torch.Tensor of size (n_levels, n_particles, *data_shape_ones) or (n_particles, *data_shape_ones)): Step size for each noise level
        * reweight_threshold (float): ESS threshold for reweighting (default is 1.0)
        * target_acceptance (float): Target acceptance rate for local steps (default is 0.75)
        * return_intermediates (bool): Whether to return all the intermediates samples (default is False)
        * samples_device (torch.device or str): Device on which to allocate the samples buffer; pass 'cpu' to keep
            the trajectory off the compute device (default is None = same device as x_init)
        * verbose (bool): Whether to display a progress bar (default is True)

    Returns:
        if return_intermediates:
            * samples (torch.Tensor of shape (n_levels, n_mcmc_steps, n_particles, *data_shape)): Samples at each noise level
        else
            * samples (torch.Tensor of shape (n_mcmc_steps, n_particles, *data_shape)): Samples at the last noise level
        * weights (torch.Tensor of shape (n_particles,)): Importance weights at the 0 level
        * step_sizes_per_noise (torch.Tensor of size (n_levels, n_particles, *data_shape_ones) or (n_particles, *data_shape_ones)): Updated step size for each noise level
        * log_z (torch.Tensor of shape ()): SMC log normalizing constant estimate at the last noise level
        * diags (dict): Dictionnary of diagnostics
    """
    # Parse the initial point shape
    n_particles = x_init.shape[0]
    data_shape = x_init.shape[1:]
    # Initialize the storage for the weights, gradiens and points
    # if reweight_threshold > 0.0:
    log_weights = torch.zeros((n_particles,), device=x_init.device, dtype=x_init.dtype)
    log_prob_x_prev = torch.empty((n_particles,), device=x_init.device, dtype=x_init.dtype)
    ess_logs = torch.ones((times.shape[0],))
    # Initialize the log normalizing constant estimate
    log_z = torch.zeros((), device=x_init.device, dtype=x_init.dtype)
    log_n = math.log(n_particles)
    # Initialize the sample buffer
    samples_device = x_init.device if samples_device is None else torch.device(samples_device)
    if return_intermediates:
        samples = torch.empty((times.shape[0], n_mcmc_steps, n_particles, *data_shape), device=samples_device, dtype=x_init.dtype)
    else:
        samples = torch.empty((n_mcmc_steps, n_particles, *data_shape), device=samples_device, dtype=x_init.dtype)
    mean_accs = torch.empty((times.shape[0],))
    x = x_init.clone()
    if verbose:
        r = trange(times.shape[0]-1, -1, -1)
    else:
        r = range(times.shape[0]-1, -1, -1)
    for time_id in r:
        # Set the first point
        x = x.clone()
        # Define the current log-prob and score

        def cur_log_prob_and_grad(y):
            return log_prob_and_grads(times[time_id], y)
        # Select the current step size and pre-conditioning
        if step_sizes_per_noise.shape == times.shape:
            cur_step_size = step_sizes_per_noise[time_id]
        else:
            cur_step_size = step_sizes_per_noise
        # Compute the inital values
        log_prob_x, grad_x = cur_log_prob_and_grad(x)
        # Compute the importance weights
        if (time_id != times.shape[0]-1):
            with torch.no_grad():
                log_weights += log_prob_x - log_prob_x_prev
            # Perform ressampling
            if reweight_threshold > 0.0:
                # Compute the ESS
                weights = torch.nn.functional.softmax(log_weights, dim=0)
                ess = (1.0 / torch.sum(torch.square(weights))) / n_particles
                ess_logs[time_id] = ess.cpu().clone()
                # Ressample the particles
                if ess < reweight_threshold:
                    log_z = log_z + (torch.logsumexp(log_weights, dim=0) - log_n)
                    idx = resampling_idx_systematic(log_weights)
                    x = x[idx]
                    log_prob_x = log_prob_x[idx]
                    grad_x = grad_x[idx]
                    log_weights.zero_()
        # Run the local sampler
        sum_acc = 0.0
        for step_id in range(n_warmup_mcmc_steps+n_mcmc_steps):
            # Perform the MCMC step
            x, log_prob_x, grad_x, log_acc = mala_step(x, log_prob_x, grad_x, cur_log_prob_and_grad, cur_step_size)
            # Store the acceptance and update the step size
            acc = torch.exp(torch.minimum(torch.zeros_like(log_acc), log_acc))
            if step_id >= n_warmup_mcmc_steps:
                sum_acc += acc
            if target_acceptance > 0.0:
                cur_step_size = heuristics_step_size(cur_step_size, log_acc, target_acceptance=target_acceptance)
            # Store the sample
            if step_id >= n_warmup_mcmc_steps:
                if return_intermediates:
                    samples[time_id, step_id - n_warmup_mcmc_steps] = x.to(samples_device)
                else:
                    if time_id == 0:
                        samples[step_id - n_warmup_mcmc_steps] = x.to(samples_device)
        # Store the mean acceptance and the step size
        mean_accs[time_id] = (sum_acc / n_mcmc_steps).mean()
        if step_sizes_per_noise.shape == times.shape:
            step_sizes_per_noise[time_id] = cur_step_size
        else:
            step_sizes_per_noise = cur_step_size
        # Display the logs
        if verbose:
            logs = {}
            logs['local_acc'] = float(mean_accs[time_id].cpu())
            if reweight_threshold > 0.0:
                logs['ess'] = float(ess_logs[time_id])
            r.set_postfix(**logs)
        # Rename the log-prob (alias is safe: next iter rebinds log_prob_x via cur_log_prob_and_grad before any mutation)
        log_prob_x_prev = log_prob_x
    # Add the final contribution to the log normalizing constant
    log_z = log_z + (torch.logsumexp(log_weights, dim=0) - log_n)
    # Return everything
    diags = {'local_acc': mean_accs}
    if reweight_threshold > 0.0:
        diags['ess'] = ess_logs
    return samples, torch.nn.functional.softmax(log_weights, dim=0), step_sizes_per_noise, log_z, diags

