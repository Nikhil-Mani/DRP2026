"""Numerical correctness checks, independent of the demonstration's convergence."""

import unittest
import torch
from sampling import run_mcmc, run_re, run_smc
from targets import BracketTwoModes, Gaussian, Target


class SamplerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(125)
        self.mean = torch.tensor([0.4, -0.7])
        self.base = Gaussian(torch.zeros(2), torch.ones(2))
        self.gaussian = Gaussian(self.mean, torch.ones(2))

    def test_target_density_and_gradient_against_torch(self):
        target = BracketTwoModes(dim=4, a=0.5, dtype=torch.float64)
        x = torch.randn(37, 4, dtype=torch.float64, requires_grad=True)
        expected_logp = target.dist.log_prob(x)
        expected_grad, = torch.autograd.grad(expected_logp.sum(), x)
        logp, grad = target.log_prob_and_grad(x)
        torch.testing.assert_close(logp, expected_logp)
        torch.testing.assert_close(grad, expected_grad)
        torch.testing.assert_close(target.weights, torch.tensor([2 / 3, 1 / 3], dtype=x.dtype))
        torch.testing.assert_close(target.variances[0], torch.linspace(0.01, 0.2, 4, dtype=x.dtype))

    def test_custom_target_autograd_inside_no_grad(self):
        target = Target(lambda x: -0.5 * x.square().sum(-1))
        x = torch.randn(13, 2)
        with torch.no_grad():
            logp, grad = target.log_prob_and_grad(x)
        torch.testing.assert_close(grad, -x)
        self.assertFalse(logp.requires_grad)
        self.assertFalse(grad.requires_grad)

    def test_local_metropolis_kernels_preserve_gaussian(self):
        # Start at stationarity: catches sign errors in drift or MH acceptance.
        for kernel, step_size in [("mala", 0.2), ("rwmh", 0.8)]:
            result = run_mcmc(self.gaussian, self.gaussian.sample((4096,)),
                              kernel=kernel, step_size=step_size,
                              n_samples=8, warmup=0, adapt_warmup=False)
            x = result["samples"][-1]
            torch.testing.assert_close(x.mean(0), self.mean, atol=0.07, rtol=0)
            torch.testing.assert_close(x.var(0), torch.ones(2), atol=0.1, rtol=0)

    def test_ula_executes_and_input_is_not_mutated(self):
        x = self.base.sample((16,))
        original = x.clone()
        result = run_mcmc(self.gaussian, x, kernel="ula", n_samples=5, warmup=2)
        self.assertEqual(tuple(result["samples"].shape), (5, 16, 2))
        self.assertIsNone(result["diagnostics"]["mean_acceptance_probability"])
        torch.testing.assert_close(x, original)

    def test_re_returns_target_replica(self):
        # Identical references make every swap accepted. All levels are stationary.
        result = run_re(self.gaussian, self.gaussian, n_chains=2048,
                        n_samples=12, warmup=4, n_levels=4, swap_every=2, step_size=0.2)
        self.assertEqual(tuple(result["samples"].shape), (12, 2048, 2))
        torch.testing.assert_close(result["samples"][-1].mean(0), self.mean, atol=0.09, rtol=0)
        self.assertAlmostEqual(result["diagnostics"]["swap_acc"].item(), 1.0)

    def test_re_with_different_endpoints_and_autograd(self):
        # A known shifted target checks both the ladder direction and target index.
        target = Target(lambda x: -0.5 * (x-self.mean).square().sum(-1))
        result = run_re(target, self.base, n_chains=1024, n_samples=128,
                        warmup=64, n_levels=8, swap_every=4, step_size=0.2)
        torch.testing.assert_close(result["samples"][-1].mean(0), self.mean, atol=0.12, rtol=0)

    def test_smc_weights_and_normalizer_without_resampling(self):
        # Known Gaussian shift with an additive log constant tests accumulated weights.
        shift = torch.tensor([1.0, -0.8])
        target = Target(lambda x: -0.5 * (x-shift).square().sum(-1)
                         - torch.log(torch.tensor(2 * torch.pi)) + 0.8)
        result = run_smc(target, self.base, n_particles=16000, n_levels=6,
                         mcmc_steps=2, step_size=0.1, resample_threshold=0.0)
        weights = result["weights"]
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=5)
        estimate = (result["samples"] * weights[:, None]).sum(0)
        torch.testing.assert_close(estimate, shift, atol=0.09, rtol=0)
        self.assertAlmostEqual(result["diagnostics"]["log_z_ratio_estimate"], 0.8, delta=0.07)

    def test_smc_resampling_and_custom_target(self):
        # A separated mean forces resampling; final weights still matter.
        shift = torch.tensor([1.8, -1.5])
        target = Target(lambda x: -0.5 * (x-shift).square().sum(-1)
                         - torch.log(torch.tensor(2 * torch.pi)))
        result = run_smc(target, self.base, n_particles=8000, n_levels=12,
                         mcmc_steps=3, step_size=0.2, resample_threshold=0.95)
        estimate = (result["samples"] * result["weights"][:, None]).sum(0)
        torch.testing.assert_close(estimate, shift, atol=0.1, rtol=0)
        self.assertLess(result["diagnostics"]["ess"].min().item(), 0.95)
        self.assertAlmostEqual(result["diagnostics"]["log_z_ratio_estimate"], 0, delta=0.12)

    def test_float64_and_seed_reproducibility(self):
        gaussian = Gaussian(torch.zeros(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64))
        torch.manual_seed(5)
        first = run_re(gaussian, gaussian, n_chains=8, n_samples=12, warmup=4,
                       n_levels=4, swap_every=2)["samples"]
        torch.manual_seed(5)
        second = run_re(gaussian, gaussian, n_chains=8, n_samples=12, warmup=4,
                        n_levels=4, swap_every=2)["samples"]
        self.assertEqual(first.dtype, torch.float64)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        result = run_smc(gaussian, gaussian, n_particles=32, n_levels=3, mcmc_steps=2)
        self.assertEqual(result["samples"].dtype, torch.float64)

    def test_bad_arguments_fail_early(self):
        with self.assertRaises(ValueError):
            run_re(self.gaussian, self.base, n_levels=2)
        with self.assertRaises(ValueError):
            run_mcmc(self.gaussian, self.base.sample((4,)), step_size=-1)
        with self.assertRaises(ValueError):
            run_smc(self.gaussian, self.base, resample_threshold=1.1)


if __name__ == "__main__":
    unittest.main()
