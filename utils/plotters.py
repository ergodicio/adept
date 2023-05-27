import os
from matplotlib import pyplot as plt
import numpy as np


def mva(actual_nk1, mod_defaults, results, td, coords):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
    ax[0].plot(coords["t"].data, actual_nk1, label="Vlasov")
    ax[0].plot(
        mod_defaults["save"]["t"]["ax"],
        (np.abs(np.fft.fft(results.ys["x"]["electron"]["n"], axis=1)[:, 1]) * 2.0 / mod_defaults["grid"]["nx"]),
        label="NN + Fluid",
    )
    ax[1].semilogy(coords["t"].data, actual_nk1, label="Vlasov")
    ax[1].semilogy(
        mod_defaults["save"]["t"]["ax"],
        (np.abs(np.fft.fft(results.ys["x"]["electron"]["n"], axis=1)[:, 1]) * 2.0 / mod_defaults["grid"]["nx"]),
        label="NN + Fluid",
    )
    ax[0].set_xlabel(r"t ($\omega_p^{-1}$)", fontsize=12)
    ax[1].set_xlabel(r"t ($\omega_p^{-1}$)", fontsize=12)
    ax[0].set_ylabel(r"$|\hat{n}|^{1}$", fontsize=12)
    ax[0].grid()
    ax[1].grid()
    ax[0].legend(fontsize=14)
    fig.savefig(os.path.join(td, "plots", "vlasov_v_fluid.png"), bbox_inches="tight")
    plt.close(fig)
