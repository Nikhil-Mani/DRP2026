import jax
from jax.core import Value
import jax.numpy as jnp
import distrax
from constants import thresholds


class Target:

    def __init__(self, a, d, alpha=1/3, cov=None):
        self.a = a
        self.d = d
        self.alpha = alpha
        if cov is None:
            self.cov = jnp.eye(self.d)
        else:
            self.cov = cov
        self.dist1 = distrax.MultivariateNormalFullCovariance(-self.a * jnp.array([1] * self.d), self.cov)
        self.dist2 = distrax.MultivariateNormalFullCovariance(self.a * jnp.array([1] * self.d), self.cov)
        self.dist = distrax.MixtureOfTwo(self.alpha, self.dist1, self.dist2)
        self.thresholds = thresholds
        self.check_seperation(self.thresholds.xi)

    def sample(self, seed=67):
        return self.dist.sample(seed=seed)
    def target(self, x):
        return self.dist.prob(x)
    def gradient(self, x):
        return jax.grad(self.dist.log_prob)(x)

    def check_seperation(self, xi):
        # Since we have isotropic case for now, we can just project the distribution onto
        # mean seperation vector
        v = jnp.asarray(self.dist1.mean()) - jnp.asarray(self.dist2.mean())
        v = jnp.linalg.vector_norm(v)
        m1 = v.T @ self.dist1.mean()
        m2 = v.T @ self.dist2.mean()
        proj_d1 = distrax.Normal(m1, jnp.sqrt(v.T @ self.dist1.covariance() @ v))
        proj_d2 = distrax.Normal(m2, jnp.sqrt(v.T @ self.dist2.covariance() @ v))

        # now calculate OVL integral:
        # point where they overlap is (m1+m2)/2
        c = (m1 + m2) / 2

        ovl = self.dist1.cdf(c) + (1 - self.dist2.cdf(c))
        if(ovl < xi):
            raise ValueError("Seperation is not enough")
