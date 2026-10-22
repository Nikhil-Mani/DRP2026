import numpy as np
import adaptivestepsize
from scipy.stats import multivariate_normal
from target import Target

def MALA(target: Target, x0, l0, total_steps, dimension):
    x = x0
    step = l0
    acceptances = []
    for i in range(total_steps):
        Z = np.random.multivariate_normal(np.zeros(dimension), np.identity(dimension))
        step = adaptivestepsize.compute_step_size(step, acceptances, total_steps)
        xp = x + step * target.gradient(x) + np.sqrt(2 * step) * Z

        covariance = 2 * step * np.eye(dimension, dtype=np.float64)
        numerator = target.target(xp) * multivariate_normal.pdf(
                                    x,
                                    mean=xp + step * target.gradient(xp),
                                    cov=covariance # pyright: ignore[reportArgumentType]
        )
        denominator = target.target(x) * multivariate_normal.pdf(
                                        xp,
                                        mean = x + step * target.gradient(x),
                                        cov=covariance # pyright: ignore[reportArgumentType]
        )
        p_accept = min(1, numerator / denominator)
        if (np.random.uniform() <= p_accept):
            acceptances.append(1)
            x = xp
        else:
            acceptances.append(0)
    return x
