import numpy as np


def compute_step_size(step_size, recent_acceptances, t):
    """
    Adapt the MALA step size using the acceptance rate.

    Parameters
    ----------
    step_size : float
        Current MALA step size (Delta t).

    recent_acceptances : list
        List of accepted (1) or rejected (0) proposals.

    t : int
        Current adaptation iteration, starting at 1.

    Returns
    -------
    float
        Updated MALA step size.
    """

    if step_size <= 0:
        raise ValueError("Step size must be positive.")

    if t < 1:
        raise ValueError("Iteration t must be at least 1.")

    if len(recent_acceptances) == 0:
        raise ValueError("Acceptance history cannot be empty.")

    # Average acceptance rate over the last 100 proposals
    alpha = np.mean(recent_acceptances[-100:])

    # Acceptance rate thresholds from the paper
    acc_low = 0.3
    acc_up = 0.6

    # Adaptation rate
    learning_rate = 1 / np.sqrt(t)

    # Too many proposals accepted: increase step size
    if alpha > acc_up:
        step_size *= np.exp(learning_rate * (alpha - acc_up))

    # Too few proposals accepted: decrease step size
    elif alpha < acc_low:
        step_size *= np.exp(learning_rate * (alpha - acc_low))

    # Otherwise, leave step size unchanged
    return float(step_size)


# ==================================
# Example: Test the function
# ==================================

if __name__ == "__main__":

    initial_step_size = 1.0

    # 80 accepted, 20 rejected
    recent_acceptances = [1] * 80 + [0] * 20

    iteration = 100

    new_step_size = compute_step_size(
        initial_step_size,
        recent_acceptances,
        iteration
    )

    print("Original step size:", initial_step_size)
    print("Acceptance rate:", np.mean(recent_acceptances))
    print("Updated step size:", new_step_size)