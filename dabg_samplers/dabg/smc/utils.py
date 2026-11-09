# Usefull function when running SMC

# Libraries
import torch


def resampling_idx_systematic(log_weights):
    """Compute the systematic resampling indices

    Args:
        * log_weights (torch.Tensor of shape (n_particles,)): Log-weights

    Returns:
        * idx (torch.Tensor of shape (n_particles,)): Resampling indices
    """
    n_particles = len(log_weights)
    weights = torch.nn.functional.softmax(log_weights, dim=0)
    cum_weights = torch.cumsum(weights, dim=0)
    u = torch.rand(1, device=log_weights.device) / n_particles
    idx = torch.searchsorted(cum_weights, u + torch.arange(n_particles, device=log_weights.device) / n_particles)
    # Guard against FP drift: cum_weights[-1] may be slightly below 1.0, pushing searchsorted to n_particles
    return idx.clamp_(max=n_particles-1)


def log_prob_tempering(t, log_prob1, log_prob2, T):
    return (t / T).flatten() * log_prob1 + (1. - (t / T)).flatten() * log_prob2


def grad_tempering(t, grad1, grad2, T):
    return (t / T) * grad1 + (1. - (t / T)) * grad2


def getter_log_prob_and_grads_tempering(dist1, dist2, T=1.0):
    """Get the log_prob_and_grads for the tempering distribution"""
    def log_prob_and_grads_tempering(t, x):
        log_prob1, grad1 = dist1.log_prob_and_grad(x)
        log_prob2, grad2 = dist2.log_prob_and_grad(x)
        log_prob = log_prob_tempering(t, log_prob1, log_prob2, T)
        grad = grad_tempering(t, grad1, grad2, T)
        return log_prob, grad
    return log_prob_and_grads_tempering

