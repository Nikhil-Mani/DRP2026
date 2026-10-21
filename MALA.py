import numpy as np
import adaptivestepsize
from scipy.stats import multivariate_normal as mvn

def gradient(target, x):
    return 17

def MALA(target, x0, l0, total_steps, dimension):
    x = x0
    step = l0
    acceptances = []
    for i in range(total_steps):
        Z = np.random.multivariate_normal(np.zeros(dimension), np.identity(dimension))
        step = adaptivestepsize.compute_step_size(step, acceptances, total_steps)
        xp = x + step * gradient(target, x) + np.sqrt(2 * step) * Z

        covariance = 2 * step * np.eye(dimension, dtype=np.float64)
        numerator = target(xp) * mvn.pdf(
                                    x,
                                    mean=xp + step * gradient(target, xp), 
                                    cov=covariance # pyright: ignore[reportArgumentType]
        )
        denominator = target(x) * mvn.pdf(
                                        xp,
                                        mean = x + step * gradient(target, x),
                                        cov=covariance # pyright: ignore[reportArgumentType]
        )
        p_accept = np.min(1, numerator / denominator)
        if (np.random.uniform() <= p_accept):
            x = xp
    return x

