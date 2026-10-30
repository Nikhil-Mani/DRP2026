import numpy as np
from target import Target
import scipy
import MALA 

def update_distribution(target: Target, init_distribution, beta):
    return lambda x: (
        init_distribution.prob(x) ** (1 - beta)
        * target.target(x) ** beta
    )

def compute_delta():
    return 17

# Parameters:
#   target - target distribution
#   num_particles - number of particles
#   init_distribution - initial distribution to sample from
#   ESS_min - minimum effective sample size
#   max_steps - maximum allowed iterations
def SMC(target: Target, num_particles, init_distribution, ESS_min, max_steps, dimension):
    beta = 0
    steps = 0
    weights = (1 / num_particles) * np.ones(num_particles)
    particles = np.ones(num_particles)
    distribution = init_distribution
    for i in range(num_particles):
        particles[i] = init_distribution.sample()

    while (beta < 1 and steps < max_steps):
        delta = compute_delta()       
        beta += delta 
        next_distribution = update_distribution(target, init_distribution, beta) 
        
        for i in range(num_particles):
            weights[i] = next_distribution(particles[i]) / distribution(particles[i])
        weights = weights / np.sum(weights)

        samples = np.random.multinomial(num_particles, weights, num_particles)
        particles = particles[samples]

        # TODO: replace these parameters
        mala_step_size = 0.5
        total_steps = 1000

        for i in range(num_particles):
            particles[i] = MALA.MALA(target, particles[i], mala_step_size, total_steps, dimension)

    return particles