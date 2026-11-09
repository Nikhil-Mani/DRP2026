# Classic Replica Exchange implementation

# Libraries
import torch
from tqdm import trange
from .utils import make_re_pairings, online_logsumexp_update, finalize_log_z
from ..mcmc.utils import heuristics_step_size
from ..mcmc.samplers import mala_step


def re_step(x, log_prob_x, grad_x, log_prob_and_grads, times, idx_i, idx_j, batch_size, data_shape, data_shape_ones,
            swap_batch_evals=True):
    """Make a swap step

    Args:
        * x (torch.Tensor of shape (n_levels, batch_size, *data_shape)): Input samples (at each noise level)
        * log_prob_x (torch.Tensor of shape (n_levels, batch_size,)): Log-prob of the input samples at each noise level
        * grad_x (torch.Tensor of shape (n_levels, batch_size, *data_shape)): Score of the input samples at each noise level
        * log_prob_and_grads (function taking t of shape (K, batch_size, *data_shape_ones) and x of shape (K, batch_size, *data_shape)): Intermediate distributions
        * times (torch.Tensor of shape (n_levels, batch_size, *data_shape_ones)): Intermediate times
        * idx_i, idx_j (torch.Tensor of shape (n_pairs,)): Indexes of the different pairs
        * batch_size (int): Batch size
        * data_shape (tuple of int): Shape of the data
        * data_shape_ones (tuple of 1): (1,) * len(data_shape)
        * swap_batch_evals (bool): Fuse the two cross-level log_prob_and_grads evaluations into one batched call
            (halves call overhead but doubles peak activation memory) (default is True)
    Returns:
        * x (torch.Tensor of shape (n_levels * batch_size, *data_shape)): Input samples (at each noise level)
        * log_prob_x (torch.Tensor of shape (n_levels * batch_size,)): Log-prob of the input samples at each noise level
        * grad_x (torch.Tensor of shape (n_levels * batch_size, *data_shape)): Score of the input samples at each noise level
        * re_acc (float): Mean swap acceptance
        * mask_matrix (torch.Tensor of shape (n_pairs, batch_size)): Acceptance mask matrix
        * log_w_fwd (torch.Tensor of shape (n_pairs, batch_size)): Forward FEP log-weight log[pi^j(X^i)/pi^i(X^i)] per pair
        * log_w_bwd (torch.Tensor of shape (n_pairs, batch_size)): Backward FEP log-weight log[pi^j(X^j)/pi^i(X^j)] per pair
    """
    # Get the length of the pair
    pair_length = idx_i.shape[0]
    # Cross-level log-prob and grad evaluations
    if swap_batch_evals:
        times_cat = torch.cat([times[idx_i], times[idx_j]], dim=0)
        x_cat = torch.cat([x[idx_j], x[idx_i]], dim=0)
        with torch.no_grad():
            log_prob_cat, grad_cat = log_prob_and_grads(times_cat, x_cat)
        p_i_j, p_j_i = log_prob_cat[:pair_length], log_prob_cat[pair_length:]
        grad_i_j, grad_j_i = grad_cat[:pair_length], grad_cat[pair_length:]
    else:
        with torch.no_grad():
            p_i_j, grad_i_j = log_prob_and_grads(times[idx_i], x[idx_j])
        with torch.no_grad():
            p_j_i, grad_j_i = log_prob_and_grads(times[idx_j], x[idx_i])
    # Compute the forward and backward FEP log-weights
    log_w_fwd = p_j_i - log_prob_x[idx_i]
    log_w_bwd = log_prob_x[idx_j] - p_i_j
    # Compute the log acceptance ratio
    log_acc = log_w_fwd - log_w_bwd
    # Get the mask
    # Equivalent to torch.log(torch.rand_like(log_acc)) < log_acc
    mask_matrix = torch.rand_like(log_acc).log_().lt_(log_acc).bool()
    re_acc = mask_matrix.float().mean()
    # Stash the half of the swap that gets overwritten first
    x_i_old = x[idx_i].clone()
    mask_data = mask_matrix.view((pair_length, batch_size, *data_shape_ones))
    log_prob_x[idx_i] = torch.where(mask_matrix, p_i_j, log_prob_x[idx_i])
    log_prob_x[idx_j] = torch.where(mask_matrix, p_j_i, log_prob_x[idx_j])
    x[idx_i] = torch.where(mask_data, x[idx_j], x[idx_i])
    x[idx_j] = torch.where(mask_data, x_i_old, x[idx_j])
    grad_x[idx_i] = torch.where(mask_data, grad_i_j, grad_x[idx_i])
    grad_x[idx_j] = torch.where(mask_data, grad_j_i, grad_x[idx_j])
    return x, log_prob_x, grad_x, re_acc, mask_matrix, log_w_fwd, log_w_bwd


def re_sampler(x_init, times, log_prob_and_grads, swap_frequency, n_warmup_mcmc_steps, n_mcmc_steps, step_sizes_per_noise,
               per_noise_init=False, target_acceptance=0.75, return_intermediates=False, swap_batch_evals=True,
               samples_device=None, verbose=True):
    """Replica-Exchange sampler

    Args:
        * x_init (torch.Tensor of shape (n_levels, batch_size, *data_shape) if per_noise_init and (batch_size, *data_shape) otherwise): Initial samples
        * times (torch.Tensor of shape (n_levels, batch_size, *data_shape_ones)): Value of the intermediate noise levels
        * log_prob_and_grads (function taking t of shape (batch_size, *data_shape_ones) and x of shape (batch_size, *data_shape)): Intermediate distributions
        * swap_frequency (int): Number of MCMC steps between each local step
        * n_warmup_mcmc_steps (int): Number of initial warmup steps for each noise level
        * n_mcmc_steps (int): Total number of MCMC steps (Swap + Local)
        * step_sizes_per_noise (torch.Tensor of size (n_levels, batch_size, *data_shape_ones)): Step size for each noise level
        * per_noise_init (bool): Whether x_init contains per noise initialization (default is False)
        * target_acceptance (float): Target acceptance rate for local steps (default is 0.75)
        * return_intermediates (bool): Whether to return all the intermediates samples (default is False)
        * swap_batch_evals (bool): Fuse the two cross-level log_prob_and_grads evaluations in each swap step
            (halves call overhead but doubles peak activation memory) (default is True)
        * samples_device (torch.device or str): Device on which to allocate the samples buffer; pass 'cpu' to keep
            the trajectory off the compute device (default is None = same device as x_init)
        * verbose (bool): Whether to display a progress bar (default is True)

    Returns:
        if return_intermediates:
            * samples (torch.Tensor of shape (n_levels, n_mcmc_steps, batch_size, *data_shape)): Samples at each noise level
        else
            * samples (torch.Tensor of shape (n_mcmc_steps, batch_size, *data_shape)): Samples at the last noise level
        * step_sizes_per_noise (torch.Tensor of size (n_levels, batch_sizes, *data_shape_ones)): Updated step size for each noise level
        * log_z (torch.Tensor of shape ()): RE log normalizing constant estimate log Z_{N-1}/Z_0, computed as the geometric mean of the FEP forward and backward estimators (Zhang et al. 2025, K=0 case)
        * diags (dict): Dictionnary of diagnostics
    """
    # Get the shapes
    if per_noise_init:
        batch_size = x_init.shape[1]
        data_shape = x_init.shape[2:]
    else:
        batch_size = x_init.shape[0]
        data_shape = x_init.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    # Initalize the storage
    samples_device = x_init.device if samples_device is None else torch.device(samples_device)
    if return_intermediates:
        samples = torch.empty((times.shape[0], n_mcmc_steps, batch_size, *data_shape), device=samples_device, dtype=x_init.dtype)
    else:
        samples = torch.empty((n_mcmc_steps, batch_size, *data_shape), device=samples_device, dtype=x_init.dtype)
    mean_local_accs = torch.zeros((times.shape[0],))
    mean_swap_acc = 0.0
    # Define the scores and log_probs
    time_ones = times.reshape((-1, *data_shape_ones))
    def local_log_prob_and_grads(y):
        return log_prob_and_grads(time_ones, y)

    def log_prob_and_grads_batched(t, y):
        log_prob, grad = log_prob_and_grads(t.view((-1, *data_shape_ones)), y.view((-1, *data_shape)))
        return log_prob.view(y.shape[:2]), grad.view(y.shape)
    # Get the initial point
    if per_noise_init:
        x = x_init.clone()
    else:
        x = x_init.unsqueeze(0).repeat((times.shape[0], 1, *data_shape_ones))
    x = x.view((-1, *data_shape))
    step_sizes_per_noise = step_sizes_per_noise.view((-1, *data_shape_ones))
    # Initialize log_prob_x and grad_x
    log_prob_x, grad_x = local_log_prob_and_grads(x)
    # Make the pairings
    pairs = make_re_pairings(times.shape[0], x_init.device)
    count_pairs = ([],[])
    # Streaming logsumexp accumulators for the FEP log-weights (one running max / sum per gap)
    n_gaps = times.shape[0] - 1
    log_z_fwd_max = torch.full((n_gaps,), float('-inf'), device=x_init.device, dtype=x_init.dtype)
    log_z_fwd_sum = torch.zeros((n_gaps,), device=x_init.device, dtype=x_init.dtype)
    log_z_bwd_max = torch.full((n_gaps,), float('-inf'), device=x_init.device, dtype=x_init.dtype)
    log_z_bwd_sum = torch.zeros((n_gaps,), device=x_init.device, dtype=x_init.dtype)
    log_z_count = torch.zeros((n_gaps,), dtype=torch.long, device=x_init.device)
    gap_ids_per_parity = [pairs[0][:, 0], pairs[1][:, 0]]
    # Run the algorithm
    if verbose:
        r = trange(n_warmup_mcmc_steps+n_mcmc_steps)
    else:
        r = range(n_warmup_mcmc_steps+n_mcmc_steps)
    for step_id in r:
        if step_id % swap_frequency == 0:
            # Select the pairs
            swap_id = int(step_id // swap_frequency) % 2
            # Reshape the data into (n_levels, batch_size, ...)
            x = x.view((-1, batch_size, *data_shape))
            grad_x = grad_x.view((-1, batch_size, *data_shape))
            log_prob_x = log_prob_x.view((-1, batch_size))
            # Perform the step
            x, log_prob_x, grad_x, re_acc, mask_matrix, log_w_fwd, log_w_bwd = re_step(
                x=x, log_prob_x=log_prob_x, grad_x=grad_x, log_prob_and_grads=log_prob_and_grads_batched, times=times,
                idx_i=pairs[swap_id][:, 0], idx_j=pairs[swap_id][:, 1], batch_size=batch_size, data_shape=data_shape,
                data_shape_ones=data_shape_ones, swap_batch_evals=swap_batch_evals)
            count_pairs[swap_id].append(mask_matrix.clone().float())
            # Update the per-gap streaming logsumexp accumulators after warmup
            if step_id >= n_warmup_mcmc_steps:
                gap_ids = gap_ids_per_parity[swap_id]
                online_logsumexp_update(log_z_fwd_max, log_z_fwd_sum, log_w_fwd.detach(), gap_ids)
                online_logsumexp_update(log_z_bwd_max, log_z_bwd_sum, -log_w_bwd.detach(), gap_ids)
                log_z_count[gap_ids] += batch_size
            # Reshape the data into (n_levels * batch_size, ....)
            x = x.view((-1, *data_shape))
            grad_x = grad_x.view((-1, *data_shape))
            log_prob_x = log_prob_x.flatten()
            mean_swap_acc = re_acc
        else:
            # Perform the local step
            x, log_prob_x, grad_x, log_acc = mala_step(
                x, log_prob_x, grad_x, local_log_prob_and_grads, step_sizes_per_noise)
            # Adapt the step-size
            if target_acceptance > 0.0:
                step_sizes_per_noise = heuristics_step_size(
                    step_sizes_per_noise, log_acc, target_acceptance=target_acceptance)
            # Log the acceptance
            acc = torch.exp(torch.minimum(torch.zeros_like(log_acc), log_acc))
            mean_local_accs = acc.view((-1, batch_size)).mean(dim=-1)
        # Store the samples
        if step_id >= n_warmup_mcmc_steps:
            if return_intermediates:
                samples[:, step_id-n_warmup_mcmc_steps] = x.reshape((-1, batch_size, *data_shape)).to(samples_device)
            else:
                samples[step_id-n_warmup_mcmc_steps] = x.view((-1, batch_size, *data_shape))[0].to(samples_device)
        if verbose:
            diags = {
                'swap_acc': mean_swap_acc.item(),
                'local_acc': mean_local_accs.mean().item()
            }
            r.set_postfix(**diags)
    # Compute the log normalizing constant estimate (paper geometric mean of forward and backward FEP)
    log_z = finalize_log_z(log_z_fwd_max, log_z_fwd_sum, log_z_bwd_max, log_z_bwd_sum, log_z_count, x_init.device)
    # Return the samples
    final_diags = {
        'swap_acc': mean_swap_acc,
        'local_acc': mean_local_accs,
        'accepted_pairs' : {
            'even' : torch.stack(count_pairs[0]).cumsum(dim=0).mean(dim=-1),
            'odd' : torch.stack(count_pairs[1]).cumsum(dim=0).mean(dim=-1),
        }
    }
    return samples, step_sizes_per_noise, log_z, final_diags

