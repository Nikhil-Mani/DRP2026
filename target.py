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

    def sample_from_hessian(self, key, mu, num, shape=() ) -> jax.Array:
        if num == 1:
            d = self.dist1
        else:
            d = self.dist2
        hess_fn = jax.hessian(d.log_prob)
        H = hess_fn(mu)

        covariance_matrix = jnp.linalg.inv(-H)

        samples = jax.random.multivariate_normal(
            key=key,
            mean=mu,
            cov=covariance_matrix,
            shape=shape
        )

        return samples

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
        #
        peak_log_prob = max(projected.log_prob(m1), projected.log_prob(m2))

        accept_ratio = jnp.exp(valley.fun - peak_log_prob)
        if (accept_ratio >= threshold):
            raise ValueError("Separation is not enough")

        # now calculate OVL integral:
        # point where they overlap is (m1+m2)/2
        # c = (m1 + m2) / 2

        # ovl = self.dist1.cdf(c) + (1 - self.dist2.cdf(c))
        # if(ovl < xi):
        #     raise ValueError("Separation is not enough")


# TODO: review this code from Gemini
class TemperedTarget:
    """A wrapper that provides a .dist.log_prob interface returning a pure scalar for JAX autodiff."""
    def __init__(self, target, init_distribution, beta):
        self.target_obj = target
        self.init_dist = init_distribution
        self.beta = beta
        self.d = target.d
        
        class TemperedDist:
            def __init__(self, parent):
                self.parent = parent
            
            def log_prob(self, x):
                mean = self.parent.init_dist.mean
                std = self.parent.init_dist.std
                variance = std ** 2
                
                # jnp.sum ensures scalar output shape () even if x has shape (1,)
                log_p0 = jnp.sum(-0.5 * jnp.log(2 * jnp.pi * variance) - jnp.square(x - mean) / (2 * variance))
                log_target = jnp.sum(self.parent.target_obj.dist.log_prob(x))
                
                return (1.0 - self.parent.beta) * log_p0 + self.parent.beta * log_target

        self.dist = TemperedDist(self)

    def target(self, x):
        particle_input = jnp.atleast_1d(x)
        t_val = float(self.target_obj.target(particle_input))
        p_val = float(self.init_dist.prob(x))
        t_val = max(t_val, 1e-300)
        p_val = max(p_val, 1e-300)
        return (p_val ** (1.0 - self.beta)) * (t_val ** self.beta)