import numpy as np
from MALA import MALA
from SMC import SMC
from constants import MALAParams, SMCParams
from target import Target
from plot_history import plot_history
import matplotlib.pyplot as plt

a = [0.5, 2.875, 5.25, 7.625]
d = [4, 8, 16, 32]

t = Target(a[1], d[0])

def test_mala():
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
def test_smc(smcparams: SMCParams, malaparams: MALAParams):
    # first initialize the mala step size
    # need to decide whether to do two seperate initializations or just one

    # create initial distribution p0
    p0 = None
    min_ess = None
    SMC(t, smcparams.num_particles, p0, min_ess, smcparams.max_steps, malaparams)

def main():
    s_params = SMCParams(0, 0)
    m_params = MALAParams(0, 0)
    test_mala()
    test_smc(s_params, m_params)

if __name__ == "__main__":
    main()
