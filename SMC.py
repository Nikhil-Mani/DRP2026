import numpy as np
from target import Target 
from target import TemperedTarget
import scipy
import MALA
from constants import *
import jax.numpy as jnp
    
def update_distribution(target: Target, init_distribution, beta):
    def log_prob_fn(x):
        particle_input = jnp.atleast_1d(x)
        t_val = float(target.target(particle_input))
        p_val = float(init_distribution.prob(x))
        
        # Guard against zero probabilities before taking logs
        log_t = np.log(max(t_val, 1e-300))
        log_p = np.log(max(p_val, 1e-300))
        
        return (1.0 - beta) * log_p + beta * log_t
        
    return log_prob_fn


def compute_delta(target: Target, particles, init_distribution, beta, ESS_min):
    num_particles = len(particles)
    remaining_beta = 1.0 - beta

    log_ratios = np.zeros(num_particles)
    for i in range(num_particles):
        particle_input = jnp.atleast_1d(particles[i])
        t_val = float(target.target(particle_input))
        p_val = float(init_distribution.prob(particles[i])) # Fixed: correctly call prob()
        
        # Guard against zero probabilities
        t_val = max(t_val, 1e-300)
        p_val = max(p_val, 1e-300)
        
        log_ratios[i] = np.log(t_val) - np.log(p_val)

    def get_ess(delta):
        if delta <= 0:
            return num_particles
        log_w = delta * log_ratios
        max_log_w = np.max(log_w)
        w = np.exp(log_w - max_log_w)
        sum_w = np.sum(w)
        sum_w_sq = np.sum(w ** 2)
        return (sum_w ** 2) / sum_w_sq

    if get_ess(remaining_beta) >= ESS_min:
        return remaining_beta, log_ratios # Return log_ratios to reuse them!

    try:
        delta_opt = scipy.optimize.brentq(
            lambda d: get_ess(d) - ESS_min,
            a=0.0,
            b=remaining_beta,
            xtol=1e-6
        )
        return delta_opt, log_ratios
    except ValueError:
        return 1e-4, log_ratios

# Parameters:
#   target - target distribution
#   num_particles - number of particles
#   init_distribution - initial distribution to sample from
#   ESS_min - minimum effective sample size
#   max_steps - maximum allowed iterations
def SMC(target: Target, init_distribution, smc_params: SMCParams, mala_params: MALAParams):
    num_particles = smc_params.num_particles
    max_steps = smc_params.max_steps
    ESS_min = smc_params.ESS_alpha * num_particles

    dimension = target.d
    beta = 0.0
    steps = 0
    
    particles = np.array([init_distribution.sample() for _ in range(num_particles)])

    while (beta < 1.0 and steps < max_steps):
        delta, log_ratios = compute_delta(target, particles, init_distribution, beta, ESS_min)
        beta += delta
        steps += 1

        log_weights = delta * log_ratios
        max_log_w = np.max(log_weights)
        weights = np.exp(log_weights - max_log_w)
        
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            weights = np.ones(num_particles) / num_particles

        samples = np.random.multinomial(num_particles, weights)
        particles = particles[samples]

        mala_step_size = mala_params.step_size
        total_steps = mala_params.total_steps

        # Create the tempered target for the current beta
        tempered_target = TemperedTarget(target, init_distribution, beta)

        # Mutation step via MALA targeting the intermediate distribution pi_beta
        for i in range(num_particles):
            particle_vector = jnp.reshape(jnp.asarray(particles[i]), (dimension,))
            particles[i] = MALA.MALA(tempered_target, particle_vector, mala_step_size, total_steps, dimension)[0]

    return particles