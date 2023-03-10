from scipy import signal
import numpy as np
from jax import numpy as jnp
from es1d import pushers
import haiku as hk


def get_nlfs(ek, dt):
    """
    Calculate the shift in frequency with respect to a reference
    This can be done by subtracting a signal at the reference frequency from the
    given signal
    :param ek:
    :param dt:
    :return:
    """

    midpt = int(ek.shape[0] / 2)

    window = 1
    # Calculate hilbert transform
    analytic_signal = signal.hilbert(window * np.real(ek))
    # Determine envelope
    amplitude_envelope = np.abs(analytic_signal)
    # Phase = angle(signal)    ---- needs unwrapping because of periodicity
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    # f(t) = dphase/dt
    instantaneous_frequency = np.gradient(instantaneous_phase, dt)  ### Sampling rate!
    # delta_f(t) = f(t) - driver_freq

    # Smooth the answer
    b, a = signal.butter(8, 0.125)
    instantaneous_frequency_smooth = signal.filtfilt(b, a, instantaneous_frequency, padlen=midpt)

    return amplitude_envelope, instantaneous_frequency_smooth


def get_derived_quantities(cfg):
    cfg["dx"] = cfg["xmax"] / cfg["nx"]
    cfg["dt"] = 0.7 * cfg["dx"] / 10

    cfg["nt"] = int(cfg["tmax"] / cfg["dt"] + 1)
    cfg["tmax"] = cfg["dt"] * cfg["nt"]

    return cfg


def get_array_quantities(cfg):
    cfg = {
        **cfg,
        **{
            "x": jnp.linspace(cfg["dx"] / 2, cfg["xmax"] - cfg["dx"] / 2, cfg["nx"]),
            "t": jnp.linspace(0, cfg["tmax"], cfg["nt"]),
            "t_save": jnp.linspace(0, cfg["tmax"], cfg["nt"] // cfg["save_skip"]),
            "kx": jnp.fft.fftfreq(cfg["nx"], d=cfg["dx"]) * 2.0 * np.pi,
            "kxr": jnp.fft.rfftfreq(cfg["nx"], d=cfg["dx"]) * 2.0 * np.pi,
        },
    }

    one_over_kx = np.zeros_like(cfg["kx"])
    one_over_kx[1:] = 1.0 / cfg["kx"][1:]
    cfg["one_over_kx"] = jnp.array(one_over_kx)

    return cfg


def init_state(cfg):
    n = jnp.ones(cfg["nx"])
    p = cfg["T0"] * jnp.ones(cfg["nx"])
    u = jnp.zeros(cfg["nx"])
    # e = jnp.zeros(cfg["nx"])

    return {"n": n, "u": u, "p": p}  # , "e": e}


def get_vector_field(cfg):
    def push_everything(t, y, args):
        push_n = pushers.DensityStepper(cfg["kx"])
        push_u = pushers.VelocityStepper(cfg["kx"], cfg["kxr"])
        push_e = pushers.EnergyStepper(cfg["kx"], cfg["gamma"])
        push_driver = pushers.Driver(cfg["x"])
        poisson_solver = pushers.PoissonSolver(cfg["one_over_kx"])

        n = y["n"]
        u = y["u"]
        p = y["p"]
        # e = y["e"]
        e = poisson_solver(n)
        ed = push_driver(cfg["pulse"], t)

        dn = push_n(n, u)
        du = push_u(n, u, p, e + ed)
        dp = push_e(n, u, p, e + ed)

        # de = pushers.step_e_ampere(n, u) + ed

        return {"n": dn, "u": du, "p": dp}

    return push_everything
    # return hk.without_apply_rng(hk.transform(push_everything))
