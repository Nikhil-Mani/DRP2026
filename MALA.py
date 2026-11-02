import numpy as np
import jax
import adaptivestepsize
from scipy.stats import multivariate_normal
from target import Target

def MALA(target: Target, x0, l0, total_steps, dimension, ret_accs=False, ret_his=False,
         *, adapt=True, return_step=False):
    x = np.asarray(x0, dtype=float)
    if x.shape != (dimension,):
        raise ValueError("x0 must be a vector with shape (dimension,)")
    if not np.isfinite(l0) or l0 <= 0:
        raise ValueError("Initial step size must be finite and positive")
    step = l0
    acceptances = []
    history = []
    log_prob_and_grad = jax.jit(jax.value_and_grad(target.dist.log_prob))
    log_prob, gradient = log_prob_and_grad(x)
    log_prob, gradient = float(log_prob), np.asarray(gradient)
    for i in range(total_steps):
        Z = np.random.multivariate_normal(np.zeros(dimension), np.identity(dimension))
        if adapt and acceptances:
            step = adaptivestepsize.compute_step_size(step, acceptances, i)
        xp = x + step * gradient + np.sqrt(2 * step) * Z
        proposal_log_prob, proposal_gradient = log_prob_and_grad(xp)
        proposal_log_prob = float(proposal_log_prob)
        proposal_gradient = np.asarray(proposal_gradient)

        covariance = 2 * step * np.eye(dimension, dtype=np.float64)
        log_numerator = proposal_log_prob + multivariate_normal.logpdf(
                                    x,
                                    mean=xp + step * proposal_gradient,
                                    cov=covariance # pyright: ignore[reportArgumentType]
        )
        log_denominator = log_prob + multivariate_normal.logpdf(
                                        xp,
                                        mean = x + step * gradient,
                                        cov=covariance # pyright: ignore[reportArgumentType]
        )
        log_ratio = log_numerator - log_denominator
        if np.isfinite(log_ratio) and np.log(np.random.uniform()) <= min(0, log_ratio):
            acceptances.append(1)
            x = xp
            log_prob, gradient = proposal_log_prob, proposal_gradient
        else:
            acceptances.append(0)
        history.append(x.copy())
    if(ret_accs):
        result = acceptances
    elif(ret_his):
        result = history
    else:
        result = x
    return (result, step) if return_step else result
