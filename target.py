import jax
from jax.core import Value
import jax.numpy as jnp
import distrax
from constants import thresholds
from scipy.optimize import minimize_scalar

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
        self.check_separation(self.thresholds.xi)

    def sample(self, seed=67):
        return self.dist.sample(seed=seed)
    def target(self, x):
        return self.dist.prob(x)
    def gradient(self, x):
        return jax.grad(self.dist.log_prob)(x)

    def check_separation(self, threshold):
        # Since we have isotropic case for now, we can just project the distribution onto
        # mean seperation vector
        v = jnp.asarray(self.dist1.mean()) - jnp.asarray(self.dist2.mean())
        # v = jnp.linalg.vector_norm(v)
        u = v / jnp.linalg.vector_norm(v)
        
        m1 = u @ self.dist1.mean()
        m2 = u @ self.dist2.mean()
        proj_d1 = distrax.Normal(m1, jnp.sqrt(u @ self.dist1.covariance() @ u))
        proj_d2 = distrax.Normal(m2, jnp.sqrt(u @ self.dist2.covariance() @ u))
        projected = distrax.MixtureOfTwo(self.alpha, proj_d1, proj_d2)

        valley = minimize_scalar(
            lambda x: float(projected.log_prob(x)),
            bounds=(min(float(m1), float(m2)),
                    max(float(m1), float(m2))),
            method="bounded"
        )

        peak_log_prob = max(self.dist.log_prob(-self.a * jnp.array([1] * self.d)), 
                            self.dist.log_prob(self.a * jnp.array([1] * self.d)))

        accept_ratio = jnp.exp(valley.fun - peak_log_prob)
        if (accept_ratio >= threshold):
            raise ValueError("Separation is not enough")

        # now calculate OVL integral:
        # point where they overlap is (m1+m2)/2
        # c = (m1 + m2) / 2

        # ovl = self.dist1.cdf(c) + (1 - self.dist2.cdf(c))
        # if(ovl < xi):
        #     raise ValueError("Separation is not enough")
