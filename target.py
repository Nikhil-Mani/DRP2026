import jax
import jax.numpy as jnp
import distrax


class Target:

    def __init__(self, a, d, alpha=1/3, cov=None):
        self.a = a
        self.d = d
        self.alpha = alpha
        if cov is None:
            self.cov = jnp.eye(self.d)
        else:
            self.cov = cov
        dist1 = distrax.MultivariateNormalFullCovariance(-self.a * jnp.array([1] * self.d), cov)
        dist2 = distrax.MultivariateNormalFullCovariance(self.a * jnp.array([1] * self.d), cov)
        self.dist = distrax.MixtureOfTwo(self.alpha, dist1, dist2)

    def sample(self, seed=67):
        return self.dist.sample(seed=seed)
    def target(self, x):
        return self.dist.prob(x)
    def gradient(self, x):
        return jax.grad(self.dist.log_prob)(x)
