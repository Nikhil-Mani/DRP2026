import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import SMC
import constants
from target import Target

class InitDistribution:
    def __init__(self):
        self.mean = 0.0
        self.std = 5.0

    def sample(self):
        return np.random.normal(self.mean, self.std)

    def prob(self, x):
        val = (1.0 / (self.std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - self.mean) / self.std)**2)
        # .item() or np.squeeze() converts any 0-dim or 1-dim single-element array to a pure Python float
        return float(np.squeeze(val))

    
def test_smc(): 
    np.random.seed(42)
    target = Target(30, 1)
    init_dist = InitDistribution()
    mala_params = constants.MALAParams(step_size=0.0001, total_steps=128)
    smc_params = constants.SMCParams(num_particles=256, max_steps=512, ESS_alpha=0.5)

    print("Running SMC sampler...")
    final_particles = SMC.SMC(target, init_distribution=init_dist, smc_params = smc_params, mala_params=mala_params)

    # Plotting the histogram of output samples alongside the true density curve
    plt.figure(figsize=(9, 5))
    count, bins, ignored = plt.hist(final_particles, bins=40, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='SMC Particles Histogram')
    
    # Plot true target density for comparison
    x_vals = np.linspace(-30,30, 500)
    y_vals = [float(target.target(jnp.atleast_1d(x))) for x in x_vals]
    # Normalize true density for visual comparison
    y_vals = np.array(y_vals) / np.trapezoid(y_vals, x_vals)
    plt.plot(x_vals, y_vals, 'r-', linewidth=2, label='True Mixture PDF: $\\frac{1}{3}\\mathcal{N}(-30,1) + \\frac{2}{3}\\mathcal{N}(30,1)$')
    
    plt.title('SMC Samples vs True Two-Component Gaussian Mixture')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

test_smc()