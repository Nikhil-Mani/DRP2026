# MCMC samplers

# Libraries
import torch
from .utils import sample_multivariate_normal_diag, log_prob_multivariate_normal_diag


def mala_step(
    y,
    target_log_prob_y,
    target_grad_y,
    target_log_prob_and_grad,
    step_size,
    vmap_compatible_mask=False
):
    """Step of the Metropolis-adjusted Langevin algorithm

    Args:
            y (torch.Tensor of shape (batch_size, dim)): Previous sample
            target_log_prob_y (torch.Tensor of shape (batch_size,)): Target's log-likelihood at y
            target_grad_y (torch.Tensor of shape (batch_size, dim)): Target's score at y
            target_log_prob_and_grad (function): Target log-likelihood and score
            step_size (torch.Tensor of shape (batch_size,)): Step size for each chain
            vmap_compatible_mask (bool): Use a mask which is vmap compatible (default is False)

    Returns:
            y_next (torch.Tensor of shape (batch_size, dim)): Next sample
            target_log_prob_y_next (torch.Tensor of shape (batch_size,)): Target's log-likelihood at y_next
            target_grad_y_next (torch.Tensor of shape (batch_size, dim)): Target's score at y_next
            step_size_next (torch.Tensor of shape (batch_size,)): Updated step size for each chain
            acc (float): Mean acceptance of the step
    """

    # Sample the proposal
    y_prop = sample_multivariate_normal_diag(
        batch_size=y.shape[0],
        mean=y + step_size * target_grad_y,
        variance=2.0 * step_size,
    )
    # Compute log-densities at the proposal
    target_log_prob_y_prop, target_grad_y_prop = target_log_prob_and_grad(y_prop)
    # Compute the MH ratio
    with torch.no_grad():
        joint_prop = target_log_prob_y_prop - log_prob_multivariate_normal_diag(
            y_prop,
            mean=y + step_size * target_grad_y,
            variance=2.0 * step_size,
            sum_indexes=-1,
        )
        joint_orig = target_log_prob_y - log_prob_multivariate_normal_diag(
            y,
            mean=y_prop + step_size * target_grad_y_prop,
            variance=2.0 * step_size,
            sum_indexes=-1,
        )
    # Acceptance step
    log_acc = joint_prop - joint_orig
    mask = torch.log(torch.rand_like(target_log_prob_y_prop, device=y.device)) < log_acc
    if vmap_compatible_mask:
        mask = mask.float()
        y = mask.unsqueeze(-1) * y_prop + (1.0 - mask).unsqueeze(-1) * y
        target_log_prob_y = mask * target_log_prob_y_prop + (1.0 - mask) * target_log_prob_y
        target_grad_y = mask.unsqueeze(-1) * target_grad_y_prop + (1.0 - mask).unsqueeze(-1) * target_grad_y
    else:
        y.data[mask] = y_prop[mask]
        target_log_prob_y.data[mask] = target_log_prob_y_prop[mask]
        target_grad_y.data[mask] = target_grad_y_prop[mask]
    # Return everything
    return (
        y,
        target_log_prob_y,
        target_grad_y,
        log_acc
    )


def ula_step(
    y,
    target_log_prob_y,
    target_grad_y,
    target_log_prob_and_grad,
    step_size,
):
    """Step of the Unadjusted Langevin algorithm

    Args:
            y (torch.Tensor of shape (batch_size, dim)): Previous sample
            target_log_prob_y (torch.Tensor of shape (batch_size,)): Target's log-likelihood at y
            target_grad_y (torch.Tensor of shape (batch_size, dim)): Target's score at y
            target_log_prob_and_grad (function): Target log-likelihood and score
            step_size (torch.Tensor of shape (batch_size,)): Step size for each chain

    Returns:
            y_next (torch.Tensor of shape (batch_size, dim)): Next sample
            target_log_prob_y_next (torch.Tensor of shape (batch_size,)): Target's log-likelihood at y_next
            target_grad_y_next (torch.Tensor of shape (batch_size, dim)): Target's score at y_next
    """

    # Sample the proposal
    with torch.no_grad():
        y_prop = sample_multivariate_normal_diag(
            batch_size=y.shape[0],
            mean=y + step_size * target_grad_y,
            variance=2.0 * step_size,
        )
    # Compute log-densities at the proposal
    target_log_prob_y_prop, target_grad_y_prop = target_log_prob_and_grad(y_prop)
    # Return everything
    return y_prop, target_log_prob_y_prop, target_grad_y_prop


@torch.no_grad()
def rwmh_step(
    y,
    target_log_prob_y,
    target_log_prob,
    step_size,
):
    """Step of the Random-Walk Metropolis Hasting algorithm

    Args:
            y (torch.Tensor of shape (batch_size, dim)): Previous sample
            target_log_prob_y (torch.Tensor of shape (batch_size,)): Target's log-likelihood at y
            target_log_prob (function): Target log-likelihood
            step_size (torch.Tensor of shape (batch_size,)): Step size for each chain

    Returns:
            y_next (torch.Tensor of shape (batch_size, dim)): Next sample
            target_log_prob_y_next (torch.Tensor of shape (batch_size,)): Target's log-likelihood at y_next
            step_size_next (torch.Tensor of shape (batch_size,)): Updated step size for each chain
            acc (float): Mean acceptance of the step
    """

    # Make a proposal
    with torch.no_grad():
        y_prop = y + step_size * torch.randn_like(y, device=y.device)
    # Compute the Metropolis-Hastings ratio
    target_log_prob_y_prop = target_log_prob(y_prop).flatten()
    with torch.no_grad():
        log_acc = target_log_prob_y_prop - target_log_prob_y
    mask = torch.log(torch.rand((y.shape[0],), device=y.device)) < log_acc
    y.data[mask] = y_prop[mask]
    target_log_prob_y.data[mask] = target_log_prob_y_prop[mask]
    # Return everything
    return (
        y,
        target_log_prob_y,
        log_acc
    )

