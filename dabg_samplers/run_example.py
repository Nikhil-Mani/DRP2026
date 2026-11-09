"""Run the paper's easiest separation/dimension setting on a CPU."""

import argparse
import json
from pathlib import Path
import time
import torch
from sampling import run_mcmc, run_re, run_smc
from targets import BracketTwoModes, Gaussian


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value


def make_plot(target, reference, results, filename):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharex=True, sharey=True)
    series = [("Direct reference", reference, None)] + [
        (name, result["samples"].reshape(-1, target.dim), result["weights"])
        for name, result in results.items()]
    bound = target.a + 4 * target.variances.max().sqrt().item()
    edges = np.linspace(-bound, bound, 81)
    histograms = []
    for ax, (name, samples, weights) in zip(axes, series):
        # Weighted histograms preserve the SMC weights instead of treating particles as IID.
        samples = samples.detach().cpu()
        weights = None if weights is None else weights.detach().cpu().numpy()
        hist, _, _ = np.histogram2d(samples[:, 0].numpy(), samples[:, 1].numpy(),
                                    bins=edges, weights=weights, density=True)
        histograms.append(hist)
        ax.scatter(target.means[:, 0].cpu(), target.means[:, 1].cpu(),
                   marker="x", color="#D95F02", s=50)
        ax.set(title=name, xlabel="Coordinate 1", aspect="equal")
    axes[0].set_ylabel("Coordinate 2")
    maximum = max(hist.max() for hist in histograms)
    for ax, hist in zip(axes, histograms):
        mesh = ax.pcolormesh(edges, edges, hist.T, cmap="Blues", vmin=0, vmax=maximum,
                             rasterized=True, zorder=0)
    fig.suptitle(f"First two coordinates of the {target.dim}D target · a={target.a}")
    fig.subplots_adjust(left=0.05, right=0.91, top=0.80, bottom=0.15, wspace=0.12)
    colorbar_axis = fig.add_axes([0.93, 0.2, 0.012, 0.55])
    fig.colorbar(mesh, cax=colorbar_axis, label="Probability density")
    fig.savefig(filename, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda; CPU is the default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--quick", action="store_true", help="Short execution check, not a convergence test")
    parser.add_argument("--output", type=Path, default=Path("example_results"))
    args = parser.parse_args()
    if args.dim < 2:
        parser.error("This plotting example requires --dim >= 2.")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    target = BracketTwoModes(dim=args.dim, a=args.a, device=args.device)
    # Match the paper: zero-mean Gaussian base with the target's marginal variances.
    # No target samples are used to construct this base or initialize RE/SMC.
    base = Gaussian(torch.zeros(args.dim, device=args.device), target.dist.variance)
    args.output.mkdir(parents=True, exist_ok=True)
    reference = target.sample((50000,))
    reference_mass = target.first_mode_mass(reference)
    n_samples, warmup = (128, 64) if args.quick else (2000, 500)
    n_particles = 128 if args.quick else 1024
    n_levels = 8 if args.quick else 64
    tasks = {
        "MALA": lambda: run_mcmc(target, target.means[0].repeat(16, 1),
                                 n_samples=n_samples, warmup=warmup),
        "RE": lambda: run_re(target, base, n_samples=n_samples, warmup=warmup,
                              n_levels=8 if args.quick else 32),
        "SMC": lambda: run_smc(target, base, n_particles=n_particles,
                                n_levels=n_levels, mcmc_steps=4 if args.quick else 16)}
    summary = {"seed": args.seed, "dim": args.dim, "a": args.a, "device": args.device,
               "quick": args.quick, "component_weight": 2 / 3,
               "direct_reference_mode_mass": reference_mass,
               "reference_n": len(reference), "results": {}}
    results = {}
    print(f"Target: BracketTwoModes(dim={args.dim}, a={args.a}); device={args.device}")
    print(f"Heavier component weight: {2 / 3:.4f}; direct reference region mass: {reference_mass:.4f}")
    print("Sampler       Mode mass      Error vs reference      Time")
    for name, task in tasks.items():
        # Synchronize around timings for asynchronous CUDA operations.
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = task()
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        mass = target.first_mode_mass(result["samples"], result["weights"])
        results[name] = result
        summary["results"][name] = {
            "first_mode_mass": mass, "absolute_error_vs_reference": abs(mass - reference_mass),
            "seconds": elapsed, "samples_shape": list(result["samples"].shape),
            "diagnostics": jsonable(result["diagnostics"])}
        print(f"{name:10s}    {mass:.4f}          {abs(mass-reference_mass):.4f}               {elapsed:.2f}s")
    for result in results.values():
        result["samples"] = result["samples"].detach().cpu()
        if result["weights"] is not None:
            result["weights"] = result["weights"].detach().cpu()
        result["diagnostics"] = jsonable(result["diagnostics"])
    torch.save(results, args.output / "samples.pt")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_plot(target, reference, results, args.output / "comparison.png")
    print(f"Saved samples, summary and plot to {args.output.resolve()}")
    print("These are finite-run diagnostics. A successful run alone does not establish convergence.")


if __name__ == "__main__":
    main()
