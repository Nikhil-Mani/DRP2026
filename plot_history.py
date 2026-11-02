"""Plot MALA traces and compare coordinate marginals with direct target samples."""

import matplotlib.pyplot as plt
import numpy as np


def plot_history(points, distribution, *, burn_in=0, bins=50,
                 reference_samples=20_000, seed=67, save_path=None):
    """Return a figure with a trace and marginal histogram per coordinate.

    points: MALA history with shape (steps, dimension), or (steps,) for 1D.
    distribution: a Target instance or a Distrax distribution supporting
        sample(seed=..., sample_shape=...).
    burn_in: number of initial steps excluded from the chain histograms.
        Traces still show all steps. Repeated states from rejections are kept.
    The target curves are histograms of independent reference samples, not
    exact densities. This works for arbitrary dimension and covariance.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points[:, None]
    if points.ndim != 2 or 0 in points.shape or not np.isfinite(points).all():
        raise ValueError("points must be a nonempty, finite (steps, dimension) array")
    if not isinstance(burn_in, (int, np.integer)) or not 0 <= burn_in < len(points):
        raise ValueError("burn_in must be an integer between 0 and len(points) - 1")
    if reference_samples < 1:
        raise ValueError("reference_samples must be positive")

    dist = getattr(distribution, "dist", distribution)
    reference = np.asarray(dist.sample(seed=seed, sample_shape=reference_samples))
    if reference.ndim == 1:
        reference = reference[:, None]
    if reference.shape != (reference_samples, points.shape[1]):
        raise ValueError("distribution samples must match the history's dimension")
    if not np.isfinite(reference).all():
        raise ValueError("distribution returned non-finite reference samples")

    dimension = points.shape[1]
    fig, axes = plt.subplots(dimension, 2, figsize=(12, 3 * dimension),
                             squeeze=False, layout="constrained")
    for coordinate, (trace_ax, hist_ax) in enumerate(axes):
        trace_ax.plot(np.arange(len(points)), points[:, coordinate],
                      linewidth=0.6, color="tab:blue")
        if burn_in:
            trace_ax.axvspan(0, burn_in, color="gray", alpha=0.2,
                            label="Discarded burn-in")
            if coordinate == 0:
                trace_ax.legend()
        trace_ax.set(xlabel="MALA iteration (0-based)",
                     ylabel=f"x[{coordinate}]", title=f"Coordinate {coordinate}: trace")

        retained = points[burn_in:, coordinate]
        edges = np.histogram_bin_edges(
            np.concatenate([retained, reference[:, coordinate]]), bins=bins)
        hist_ax.hist(retained, bins=edges, density=True, alpha=0.45,
                     label="MALA after burn-in", color="tab:blue")
        hist_ax.hist(reference[:, coordinate], bins=edges, density=True,
                     histtype="step", linewidth=1.6, color="tab:orange",
                     label="Target (direct samples)")
        hist_ax.set(xlabel=f"x[{coordinate}]", ylabel="Density",
                    title=f"Coordinate {coordinate}: marginal distribution")
        if coordinate == 0:
            hist_ax.legend()

    fig.suptitle(f"MALA history: {len(points):,} steps; "
                 f"{len(points) - burn_in:,} retained")
    if save_path is not None:
        fig.savefig(save_path, dpi=160)
    return fig
