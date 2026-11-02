import numpy as np
from MALA import MALA
from target import Target
from plot_history import plot_history
import matplotlib.pyplot as plt

a = [0.5, 2.875, 5.25, 7.625]
d = [4, 8, 16, 32]

def test_mala():
    t = Target(a[1], d[0])
    steps = 8192
    x0 = -t.a * np.ones(t.d)
    l = warm_up(t, x0)
    # we can just pick one of the modes to see that its working
    pts = MALA(t, x0, l, steps, t.d, ret_his=True, adapt=False)
    plot_history(pts, t, burn_in=100, save_path="mala_history.png")
    plt.show()


def warm_up(t: Target, x0):
    l = 0.001
    # perform 4092 warm up steps
    accs, l = MALA(t, x0, l, 4092, t.d, ret_accs=True, return_step=True)
    print(f"Warm-up acceptance: {np.mean(accs):.3f}; tuned step size: {l:.5f}")
    return l

if __name__ == "__main__":
    test_mala()
