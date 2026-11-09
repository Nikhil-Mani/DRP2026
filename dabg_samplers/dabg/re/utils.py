# Utils for replica exchange

# Libraries
import math
import torch


def online_logsumexp_update(max_acc, sum_acc, new_values, gap_ids):
    """Streaming logsumexp update for per-gap FEP log-weight accumulation

    Updates the running max and shifted exp-sum for each gap in-place by absorbing a new batch of values.
    At the end the per-gap logsumexp is ``max_acc[k] + log(sum_acc[k])`` (defined only where count > 0).

    Args:
        * max_acc (torch.Tensor of shape (n_gaps,)): Running max per gap (init at -inf)
        * sum_acc (torch.Tensor of shape (n_gaps,)): Running sum of exp(values - max) per gap (init at 0)
        * new_values (torch.Tensor of shape (n_pairs, batch_size)): Per-pair batch of new log-weights
        * gap_ids (torch.Tensor of shape (n_pairs,)): Destination gap index per pair
    """
    per_pair_max = new_values.max(dim=-1).values
    cur_max = max_acc[gap_ids]
    new_max = torch.maximum(cur_max, per_pair_max)
    new_sum = (sum_acc[gap_ids] * torch.exp(cur_max - new_max)
               + torch.exp(new_values - new_max.unsqueeze(-1)).sum(dim=-1))
    max_acc[gap_ids] = new_max
    sum_acc[gap_ids] = new_sum


def finalize_log_z(log_z_fwd_max, log_z_fwd_sum, log_z_bwd_max, log_z_bwd_sum, log_z_count, device):
    """Combine per-gap streaming-logsumexp accumulators into the RE log Z estimate

    Args:
        * log_z_fwd_max, log_z_fwd_sum (torch.Tensor of shape (n_gaps,)): Running max / shifted-sum for log_w_fwd
        * log_z_bwd_max, log_z_bwd_sum (torch.Tensor of shape (n_gaps,)): Running max / shifted-sum for -log_w_bwd
        * log_z_count (torch.Tensor of shape (n_gaps,)): Number of FEP samples accumulated per gap
        * device (torch.device): Device for the returned scalar
    Returns:
        * log_z (torch.Tensor of shape ()): Geometric mean of forward and backward FEP estimators
    """
    log_z_fwd = torch.zeros((), device=device)
    log_z_bwd = torch.zeros((), device=device)
    for k in range(log_z_count.shape[0]):
        n_k = log_z_count[k].item()
        if n_k == 0:
            continue
        log_n = math.log(n_k)
        log_z_fwd = log_z_fwd + (log_z_fwd_max[k] + torch.log(log_z_fwd_sum[k]) - log_n)
        log_z_bwd = log_z_bwd - (log_z_bwd_max[k] + torch.log(log_z_bwd_sum[k]) - log_n)
    return 0.5 * (log_z_fwd + log_z_bwd)

def make_re_pairings(num_noise_levels, device=None, return_masks=False):
    """Make the pairings for replica exchange

    Args:
        * num_noise_levels (int): Number of noise levels
        * device (torch.device): Torch device to use (default is None)
        * return_masks (bool): Whether to return the masks directly

    Returns:
        * pairings (list of torch.Tensor): List of two index arrays provided the noise levels
            to jump to in the even and in the odd case
    """

    arr = torch.arange(num_noise_levels, device=device)
    if return_masks:
        arr_bool = torch.zeros((num_noise_levels,), device=device,
            dtype=bool)
    # Even pass
    mask_a = (arr % 2 == 0) & (arr + 1 < num_noise_levels)
    if return_masks:
        mask_non_a = arr_bool.clone()
        mask_non_a[arr[mask_a] + 1] = True
        a = torch.stack([mask_a, mask_non_a], dim=-1)
    else:
        a = torch.stack([arr[mask_a], arr[mask_a] + 1], dim=-1)
    # Odd pass
    mask_b = (arr % 2 == 1) & (arr + 1 < num_noise_levels)
    if return_masks:
        mask_non_b = arr_bool.clone()
        mask_non_b[arr[mask_b] + 1] = True
        b = torch.stack([mask_b, mask_non_b], dim=-1)
    else:
        b = torch.stack([arr[mask_b], arr[mask_b] + 1], dim=-1)
    # Return everything
    return [a, b]



