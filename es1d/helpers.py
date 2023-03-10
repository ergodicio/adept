from scipy import signal
import numpy as np
from jax import numpy as jnp
from theory.electrostatic import get_roots_to_electrostatic_dispersion
from es1d import pushers


def get_nlfs(ek, dt):
    """
    Calculate the shift in frequency with respect to a reference
    This can be done by subtracting a signal at the reference frequency from the
    given signal
    :param ef:
    :param wepw:
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
    ww = get_roots_to_electrostatic_dispersion(1.0, 1.0, cfg["k0"])
    w0 = np.real(ww)
    wi = np.imag(ww)
    w0_corr = w0 / (np.sqrt(1 + 3 * cfg["k0"] ** 2.0))

    cfg["nt"] = int(cfg["tmax"] / cfg["dt"] + 1)
    cfg["dx"] = cfg["xmax"] / cfg["nx"]

    cfg["xax"] = jnp.linspace(cfg["dx"] / 2, cfg["xmax"] - cfg["dx"] / 2, cfg["nx"])
    cfg["kx"] = jnp.fft.fftfreq(cfg["nx"], d=cfg["dx"]) * 2.0 * np.pi
    cfg["one_over_kx"] = np.zeros_like(cfg["kx"])
    cfg["one_over_kx"][1:] = 1.0 / cfg["kx"][1:]
    cfg["one_over_kx"] = jnp.array(cfg["one_over_kx"])

    cfg["dx"] = cfg["xax"][2] - cfg["xax"][1]
    cfg["t"] = jnp.linspace(0.0, cfg["tmax"], cfg["nt"])

    return cfg


def init_state(cfg):
    n = jnp.ones(cfg["nx"])
    p = cfg["T0"] * jnp.ones(cfg["nx"])
    u = jnp.zeros(cfg["nx"])
    e = jnp.zeros(cfg["nx"])

    return {"n": n, "u": u, "p": p, "e": e}


def get_vector_field(cfg):
    def push_everything(t, y, args):
        n = y["n"]
        u = y["u"]
        p = y["p"]
        e = y["e"]

        ed = pushers.calc_driver(args["pulse"], t)

        dn = pushers.step_n(n, u)
        du = pushers.step_u(n, u, p, e)
        dp = pushers.step_p(n, u, p, e)

        de = pushers.step_e_ampere(n, u) + ed

        return {"n": dn, "u": du, "p": dp, "e": de}

    return push_everything
