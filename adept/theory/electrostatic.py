# MIT License
#
# Copyright (c) 2022 Ergodic LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Tuple
import numpy as np
import scipy
from scipy import signal


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


def plasma_dispersion(value):
    """
    This function leverages the Fadeeva function in scipy to calculate the Z function

    :param value:
    :return:
    """
    return scipy.special.wofz(value) * np.sqrt(np.pi) * 1j


def plasma_dispersion_prime(value):
    """
    This is a simple relation for Z-prime, which happens to be directly proportional to Z

    :param value:
    :return:
    """
    return -2.0 * (1.0 + value * plasma_dispersion(value))


def get_roots_to_electrostatic_dispersion(wp_e, vth_e, k0, maxwellian_convention_factor=2.0, initial_root_guess=None):
    """
    This function calculates the root of the plasma dispersion relation

    :param wp_e:
    :param vth_e:
    :param k0:
    :param maxwellian_convention_factor:
    :param initial_root_guess:
    :return:
    """
    from scipy import optimize

    plasma_epsilon, initial_root_guess = get_dispersion_function(
        wp_e, vth_e, k0, maxwellian_convention_factor, initial_root_guess
    )

    epsilon_root = optimize.newton(plasma_epsilon, initial_root_guess)

    return epsilon_root * k0 * vth_e * np.sqrt(maxwellian_convention_factor)


def get_dispersion_function(wp_e, vth_e, k0, maxwellian_convention_factor=2.0, initial_root_guess=None):
    """
    This function calculates the root of the plasma dispersion relation

    :param wp_e:
    :param vth_e:
    :param k0:
    :param maxwellian_convention_factor:
    :param initial_root_guess:
    :return:
    """
    if initial_root_guess is None:
        initial_root_guess = np.sqrt(wp_e**2.0 + 3 * (k0 * vth_e) ** 2.0)

    chi_e = np.power((wp_e / (vth_e * k0)), 2.0) / maxwellian_convention_factor

    def plasma_epsilon(x):
        val = 1.0 - chi_e * plasma_dispersion_prime(x)
        return val

    return plasma_epsilon, initial_root_guess


def calc_depsdw(kld):
    """
    Used for frequency shift calculations

    :param kld:
    :return:
    """
    depsdw = {}
    wax = np.linspace(1.0, 1.5, 2048)

    # Approximate epsilon
    epsilon_approx = 1 - 1.0 / wax**2.0 - 3 * 1 / wax**2.0 * (kld / wax) ** 2.0

    # Exact
    disp_fn, _ = get_dispersion_function(1.0, 1.0, kld)

    # Make array of epsilons
    disp_arr = np.array([np.real(disp_fn(w / (kld * np.sqrt(2.0)))) for w in wax])

    depsdw["exact"] = np.gradient(disp_arr, wax[2] - wax[1])
    depsdw["approx"] = np.gradient(epsilon_approx, wax[2] - wax[1])

    wr = np.real(
        get_roots_to_electrostatic_dispersion(1.0, 1.0, kld, maxwellian_convention_factor=2.0, initial_root_guess=None)
    )
    iw = np.argmin(np.abs(wax - wr))

    return depsdw["exact"][iw], depsdw["approx"][iw]


def get_complex_frequency_table(num: int, kinetic_real_epw: bool) -> Tuple[np.array, np.array, np.array]:
    """
    This function creates a table of the complex plasma frequency for $0.2 < k \lambda_D < 0.4$ in `num` steps

    :param kinetic_real_epw:
    :param num:
    :return:
    """
    klds = np.linspace(0.02, 0.4, num)
    wrs = np.zeros(num)
    wis = np.zeros(num)

    for i, kld in enumerate(klds):
        ww = get_roots_to_electrostatic_dispersion(1.0, 1.0, kld)
        if kinetic_real_epw:
            wrs[i] = np.real(ww)
        else:
            wrs[i] = np.sqrt(1.0 + 3.0 * kld**2.0)
        wis[i] = np.imag(ww)

    return wrs, wis, klds
