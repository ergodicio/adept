#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

import numpy as np
import scipy.optimize

from adept.electrostatic import plasma_dispersion_prime


def get_ion_acoustic_frequency(k, Te_over_Ti=1.0, me_over_mi=1.0/1836.0):
    """
    Calculate the ion acoustic wave frequency using the approximate dispersion relation.

    For k*lambda_De >> 1 and Te >> Ti:
    omega^2 ≈ k^2 * cs^2 where cs^2 = (Te + Ti)/mi ≈ Te/mi for Te >> Ti

    :param k: wavenumber (normalized by Debye length)
    :param Te_over_Ti: electron to ion temperature ratio
    :param me_over_mi: electron to ion mass ratio
    :return: normalized frequency
    """
    # Sound speed squared in normalized units
    cs_squared = (1.0 + 1.0/Te_over_Ti) * me_over_mi
    return k * np.sqrt(cs_squared)


def get_two_species_dispersion_function(wp_e, vth_e, wp_i, vth_i, k0):
    """
    Two-species plasma dispersion relation:
    1 = wp_e^2/k^2 * (1/vth_e^2) * Z'(w/(k*vth_e)) + wp_i^2/k^2 * (1/vth_i^2) * Z'(w/(k*vth_i))

    :param wp_e: electron plasma frequency
    :param vth_e: electron thermal velocity
    :param wp_i: ion plasma frequency
    :param vth_i: ion thermal velocity
    :param k0: wavenumber
    :return: dispersion function
    """
    chi_e = (wp_e / (k0 * vth_e))**2
    chi_i = (wp_i / (k0 * vth_i))**2

    def dispersion_function(w_norm):
        # w_norm is w/(k*vth_e)
        zeta_e = w_norm  # w/(k*vth_e)
        zeta_i = w_norm * vth_e / vth_i  # w/(k*vth_i)

        epsilon = 1.0 - chi_e * plasma_dispersion_prime(zeta_e) - chi_i * plasma_dispersion_prime(zeta_i)
        return epsilon

    return dispersion_function


def get_two_species_roots(wp_e, vth_e, wp_i, vth_i, k0, mode='electron'):
    """
    Find roots of the two-species dispersion relation.

    :param wp_e: electron plasma frequency
    :param vth_e: electron thermal velocity
    :param wp_i: ion plasma frequency
    :param vth_i: ion thermal velocity
    :param k0: wavenumber
    :param mode: 'electron' for electron plasma waves, 'ion' for ion acoustic waves
    :return: complex frequency
    """
    dispersion_fn = get_two_species_dispersion_function(wp_e, vth_e, wp_i, vth_i, k0)

    if mode == 'electron':
        # Initial guess for electron plasma wave (high frequency)
        initial_guess = np.sqrt(wp_e**2 + 3*(k0*vth_e)**2) / (k0*vth_e)
    elif mode == 'ion':
        # Initial guess for ion acoustic wave (low frequency)
        cs = np.sqrt(vth_e**2 + vth_i**2)  # approximate sound speed
        initial_guess = k0 * cs / (k0*vth_e)
    else:
        raise ValueError("mode must be 'electron' or 'ion'")

    try:
        root_norm = scipy.optimize.newton(dispersion_fn, initial_guess)
        # Convert back to physical frequency
        root = root_norm * k0 * vth_e
        return root
    except Exception:
        # If newton fails, try a different approach
        w_range = np.linspace(0.1*initial_guess, 2.0*initial_guess, 1000)
        disp_vals = np.array([dispersion_fn(w) for w in w_range])
        idx = np.argmin(np.abs(disp_vals))
        return w_range[idx] * k0 * vth_e
