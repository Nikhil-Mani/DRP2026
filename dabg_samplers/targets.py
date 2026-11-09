"""Small target objects with the interface used by the original DABG samplers."""

import math
import torch


class Target:
    """Wrap a batched log density; a normalizing constant is not required.

    log_prob_fn maps (batch, dim) -> (batch,). It must treat rows independently.
    Provide score_fn for an analytical gradient, or use PyTorch autograd.
    """

    def __init__(self, log_prob_fn, score_fn=None):
        self.log_prob_fn = log_prob_fn
        self.score_fn = score_fn

    def log_prob(self, x):
        values = self.log_prob_fn(x)
        if values.shape != x.shape[:-1]:
            raise ValueError("log_prob must return one value per input row.")
        return values

    def log_prob_and_grad(self, x):
        # RE evaluates its target inside no_grad, so enable gradients locally.
        if self.score_fn is None:
            with torch.enable_grad():
                y = x.detach().requires_grad_(True)
                values = self.log_prob(y)
                gradient, = torch.autograd.grad(values.sum(), y)
        else:
            with torch.no_grad():
                values = self.log_prob(x)
                gradient = self.score_fn(x)
        if gradient.shape != x.shape:
            raise ValueError("score must return the same shape as the input.")
        return values.detach(), gradient.detach()


class Gaussian(Target):
    """Normalized diagonal Gaussian, used as the easy reference for RE and SMC."""

    def __init__(self, mean, variance):
        if mean.ndim != 1 or variance.shape != mean.shape:
            raise ValueError("mean and variance must be vectors of the same shape.")
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all() or (variance <= 0).any():
            raise ValueError("Gaussian parameters must be finite with positive variance.")
        self.mean = mean.detach().clone()
        self.variance = variance.detach().clone()
        super().__init__(self._log_prob, self._score)

    def _log_prob(self, x):
        return -0.5 * (((x - self.mean).square() / self.variance)
                       + self.variance.log() + math.log(2 * math.pi)).sum(-1)

    def _score(self, x):
        return -(x - self.mean) / self.variance

    def sample(self, sample_shape):
        return self.mean + self.variance.sqrt() * torch.randn(
            (*sample_shape, self.mean.numel()), device=self.mean.device, dtype=self.mean.dtype)


class BracketTwoModes(Target):
    """Same density as upstream BracketTwoModes, with its default 2:1 weights.

    Adapted from h2o64/dabg, dabg/distr/gauss.py (MIT license).
    Only sampling, log density, gradient and mode assignment are needed here.
    """

    def __init__(self, dim=4, a=0.5, device="cpu", dtype=torch.float32):
        if not isinstance(dim, int) or dim < 1 or not math.isfinite(a) or a <= 0:
            raise ValueError("dim must be positive and a must be finite and positive.")
        self.dim = dim
        self.a = a
        self.means = torch.stack([
            -a * torch.ones(dim, device=device, dtype=dtype),
            +a * torch.ones(dim, device=device, dtype=dtype)])
        diagonal = torch.linspace(0.01, 0.2, dim, device=device, dtype=dtype)
        self.variances = torch.stack([diagonal, diagonal.flip(0)])
        self.weights = torch.tensor([2 / 3, 1 / 3], device=device, dtype=dtype)
        self.dist = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(self.weights),
            torch.distributions.Independent(
                torch.distributions.Normal(self.means, self.variances.sqrt()), 1))
        super().__init__(self._log_prob, self._score)

    def component_log_prob(self, x):
        """Unweighted component densities, matching the paper's mode partition."""
        return -0.5 * (((x.unsqueeze(-2) - self.means).square() / self.variances)
                       + self.variances.log() + math.log(2 * math.pi)).sum(-1)

    def _log_prob(self, x):
        return torch.logsumexp(self.component_log_prob(x) + self.weights.log(), -1)

    def _score(self, x):
        responsibilities = (self.component_log_prob(x) + self.weights.log()).softmax(-1)
        return -(responsibilities.unsqueeze(-1)
                 * (x.unsqueeze(-2) - self.means) / self.variances).sum(-2)

    def sample(self, sample_shape):
        return self.dist.sample(sample_shape)

    def first_mode(self, x):
        # Use component densities rather than mixture-weighted responsibilities.
        return self.component_log_prob(x).argmax(-1) == 0

    def first_mode_mass(self, samples, weights=None):
        indicator = self.first_mode(samples.reshape(-1, self.dim)).to(samples.dtype)
        if weights is None:
            return indicator.mean().item()
        weights = weights.reshape(-1)
        if weights.shape != indicator.shape:
            raise ValueError("There must be one weight per sample.")
        return (indicator * (weights / weights.sum())).sum().item()
