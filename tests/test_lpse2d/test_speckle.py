"""
Tests for the JAX-ported SpeckleProfile class.

Ported from: https://github.com/LASY-org/lasy/blob/development/tests/test_speckles.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest
from scipy.constants import c

from adept._lpse2d.core.speckle import SpeckleProfile


def create_2d_grid(lo, hi, num_points):
    """Create a 2D meshgrid for x, y coordinates."""
    x = jnp.linspace(lo[0], hi[0], num_points[0])
    y = jnp.linspace(lo[1], hi[1], num_points[1])
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    return X, Y


def evaluate_over_time(profile, x, y, t_array):
    """Evaluate speckle profile over multiple time points.

    Args:
        profile: SpeckleProfile instance
        x, y: 2D spatial arrays
        t_array: 1D array of times in seconds

    Returns:
        3D array of shape (nx, ny, nt)
    """

    def eval_at_t(t):
        return profile.evaluate(x, y, t)

    return jnp.stack([eval_at_t(t) for t in t_array], axis=-1)


@pytest.mark.parametrize("temporal_smoothing_type", ["RPP", "CPP", "FM SSD", "GP RPM SSD", "GP ISI"])
def test_intensity_distribution(temporal_smoothing_type):
    """Test whether the spatial intensity distribution and statistics are correct.

    The distribution should be exponential, 1/<I> exp(-I/<I>) [Michel, 9.35].
    The real and imaginary parts of the envelope [Michel, Eqn. 9.26] and
    their product [9.30] should all be 0 on average.
    """
    key = jax.random.PRNGKey(0)

    wavelength = 0.351e-6  # Laser wavelength in meters
    polarization = (1, 0)  # Linearly polarized in the x direction
    focal_length = 3.5  # m
    beam_aperture = [0.35, 0.5]  # m
    n_beamlets = [24, 32]
    relative_laser_bandwidth = 0.005
    ssd_phase_modulation_amplitude = (4.1, 4.5)
    ssd_number_color_cycles = [1.4, 1.0]
    ssd_transverse_bandwidth_distribution = [1.8, 1.0]

    dx = wavelength * focal_length / beam_aperture[0]
    dy = wavelength * focal_length / beam_aperture[1]
    Lx = 1.8 * dx * n_beamlets[0]
    Ly = 3.1 * dy * n_beamlets[1]
    nu_laser = c / wavelength
    t_max = 50 / nu_laser
    lo = (0, 0)
    hi = (Lx, Ly)
    num_points = (200, 250)
    num_t = 2

    profile = SpeckleProfile(
        wavelength,
        polarization,
        focal_length,
        beam_aperture,
        n_beamlets,
        temporal_smoothing_type=temporal_smoothing_type,
        key=key,
        t_max=t_max,
        relative_laser_bandwidth=relative_laser_bandwidth,
        ssd_phase_modulation_amplitude=ssd_phase_modulation_amplitude,
        ssd_number_color_cycles=ssd_number_color_cycles,
        ssd_transverse_bandwidth_distribution=ssd_transverse_bandwidth_distribution,
    )

    x, y = create_2d_grid(lo, hi, num_points)
    t_array = jnp.linspace(0, t_max, num_t)
    F = evaluate_over_time(profile, x, y, t_array)

    # Get spatial statistics
    # <real env> = 0 = <imag env> = <er * ei>
    e_r = jnp.real(F)
    e_i = jnp.imag(F)
    er_ei = e_r * e_i

    # Check that means are close to zero relative to standard deviation
    assert jnp.max(jnp.abs(e_r.mean(axis=(0, 1)) / e_r.std(axis=(0, 1)))) < 1.0e-1
    assert jnp.max(jnp.abs(e_i.mean(axis=(0, 1)) / e_i.std(axis=(0, 1)))) < 1.0e-1
    assert jnp.max(jnp.abs(er_ei.mean(axis=(0, 1)) / er_ei.std(axis=(0, 1)))) < 1.0e-1

    # Compare intensity distribution with expected 1/<I> exp(-I/<I>)
    env_I = jnp.abs(F) ** 2
    I_vec = env_I.flatten()
    mean_I = I_vec.mean()
    N_hist = 200
    counts, bins = jnp.histogram(I_vec, bins=N_hist, density=True)
    I_dist = 1.0 / mean_I * jnp.exp(-bins / mean_I)
    error_I_dist = jnp.max(jnp.abs(counts - I_dist[:-1]))
    assert error_I_dist < 2.0e-4


@pytest.mark.parametrize("temporal_smoothing_type", ["RPP", "CPP", "FM SSD", "GP RPM SSD", "GP ISI"])
def test_spatial_correlation(temporal_smoothing_type):
    """Tests whether the speckles have the correct shape.

    The speckle shape is measured over one period, since the spatial profile is periodic.
    The correct speckle shape for a rectangular laser, determined by the
    autocorrelation, is the product of sinc functions [Michel, Eqn. 9.16].
    """
    key = jax.random.PRNGKey(0)

    wavelength = 0.351e-6
    polarization = (1, 0)
    focal_length = 3.5
    beam_aperture = [0.35, 0.35]
    n_beamlets = [24, 32]
    relative_laser_bandwidth = 0.005

    ssd_phase_modulation_amplitude = (4.1, 4.1)
    ssd_number_color_cycles = [1.4, 1.0]
    ssd_transverse_bandwidth_distribution = [1.0, 1.0]

    dx = wavelength * focal_length / beam_aperture[0]
    dy = wavelength * focal_length / beam_aperture[1]
    Lx = dx * n_beamlets[0]
    Ly = dy * n_beamlets[1]
    nu_laser = c / wavelength
    tu = 1 / relative_laser_bandwidth / 50 / nu_laser
    t_max = 200 * tu
    lo = (0, 0)
    hi = (Lx, Ly)
    num_points = (200, 200)
    num_t = 300

    profile = SpeckleProfile(
        wavelength,
        polarization,
        focal_length,
        beam_aperture,
        n_beamlets,
        temporal_smoothing_type=temporal_smoothing_type,
        key=key,
        t_max=t_max,
        relative_laser_bandwidth=relative_laser_bandwidth,
        ssd_phase_modulation_amplitude=ssd_phase_modulation_amplitude,
        ssd_number_color_cycles=ssd_number_color_cycles,
        ssd_transverse_bandwidth_distribution=ssd_transverse_bandwidth_distribution,
    )

    x, y = create_2d_grid(lo, hi, num_points)
    t_array = jnp.linspace(0, t_max, num_t)
    F = evaluate_over_time(profile, x, y, t_array)

    # Compare speckle profile / autocorrelation
    # Compute autocorrelation using Wiener-Khinchin Theorem
    fft_abs_all = jnp.abs(jnp.fft.fft2(F, axes=(0, 1))) ** 2
    ifft_abs = jnp.abs(jnp.fft.ifft2(fft_abs_all, axes=(0, 1))) ** 2
    acorr2_3d = jnp.fft.fftshift(ifft_abs, axes=(0, 1))
    acorr2_3d_norm = acorr2_3d / jnp.max(acorr2_3d, axis=(0, 1))

    # Compare with theoretical speckle profile
    x_list = jnp.linspace(-n_beamlets[0] / 2 + 0.5, n_beamlets[0] / 2 - 0.5, num_points[0], endpoint=False)
    y_list = jnp.linspace(-n_beamlets[1] / 2 + 0.5, n_beamlets[1] / 2 - 0.5, num_points[1], endpoint=False)
    X, Y = jnp.meshgrid(x_list, y_list, indexing="ij")
    acorr_theor = jnp.sinc(X) ** 2 * jnp.sinc(Y) ** 2
    error_auto_correlation = jnp.max(jnp.abs(acorr_theor[:, :, jnp.newaxis] - acorr2_3d_norm))

    assert error_auto_correlation < 5.0e-1


@pytest.mark.parametrize("temporal_smoothing_type", ["RPP", "CPP", "FM SSD", "GP RPM SSD", "GP ISI"])
def test_sinc_zeros(temporal_smoothing_type):
    r"""Test whether the transverse sinc envelope has the correct width.

    The transverse envelope for the rectangular laser has the form
    sinc(pi*x/Delta_x) * sinc(pi*y/Delta_y) [Michel, Eqns. 9.11, 87, 94].
    """
    key = jax.random.PRNGKey(0)

    wavelength = 0.351e-6
    polarization = (1, 0)
    focal_length = 3.5
    beam_aperture = [0.35, 0.35]
    n_beamlets = [24, 48]
    relative_laser_bandwidth = 0.005
    ssd_phase_modulation_amplitude = (4.1, 4.1)
    ssd_number_color_cycles = [1.4, 1.0]
    ssd_transverse_bandwidth_distribution = [1.0, 1.0]

    dx = wavelength * focal_length / beam_aperture[0]
    dy = wavelength * focal_length / beam_aperture[1]
    Lx = dx * n_beamlets[0]
    Ly = dy * n_beamlets[1]
    nu_laser = c / wavelength
    tu = 1 / relative_laser_bandwidth / 50 / nu_laser
    t_max = 200 * tu
    lo = (-Lx, -Ly)
    hi = (Lx, Ly)
    num_points = (300, 300)
    num_t = 10

    profile = SpeckleProfile(
        wavelength,
        polarization,
        focal_length,
        beam_aperture,
        n_beamlets,
        temporal_smoothing_type=temporal_smoothing_type,
        key=key,
        t_max=t_max,
        relative_laser_bandwidth=relative_laser_bandwidth,
        ssd_phase_modulation_amplitude=ssd_phase_modulation_amplitude,
        ssd_number_color_cycles=ssd_number_color_cycles,
        ssd_transverse_bandwidth_distribution=ssd_transverse_bandwidth_distribution,
        do_include_transverse_envelope=True,
    )

    x, y = create_2d_grid(lo, hi, num_points)
    t_array = jnp.linspace(0, t_max, num_t)
    F = evaluate_over_time(profile, x, y, t_array)

    # Check that edges are near zero (sinc zeros)
    assert jnp.abs(F[0, :, :]).max() / jnp.abs(F).max() < 1.0e-8
    assert jnp.abs(F[-1, :, :]).max() / jnp.abs(F).max() < 1.0e-8
    assert jnp.abs(F[:, 0, :]).max() / jnp.abs(F).max() < 1.0e-8
    assert jnp.abs(F[:, -1, :]).max() / jnp.abs(F).max() < 1.0e-8


def test_FM_SSD_periodicity():
    """Test that FM SSD has the correct temporal frequency."""
    key = jax.random.PRNGKey(0)

    wavelength = 0.351e-6
    polarization = (1, 0)
    focal_length = 3.5
    beam_aperture = [0.35, 0.35]
    n_beamlets = [24, 32]
    temporal_smoothing_type = "FM SSD"
    relative_laser_bandwidth = 0.005

    ssd_phase_modulation_amplitude = [4.1, 4.1]
    ssd_number_color_cycles = [1.4, 1.0]
    ssd_transverse_bandwidth_distribution = [1.0, 1.0]

    nu_laser = c / wavelength
    ssd_frac = jnp.sqrt(ssd_transverse_bandwidth_distribution[0] ** 2 + ssd_transverse_bandwidth_distribution[1] ** 2)
    ssd_frac = (
        ssd_transverse_bandwidth_distribution[0] / ssd_frac,
        ssd_transverse_bandwidth_distribution[1] / ssd_frac,
    )
    phase_mod_freq = [
        relative_laser_bandwidth * sf * 0.5 / pma
        for sf, pma in zip(ssd_frac, ssd_phase_modulation_amplitude, strict=True)
    ]
    t_max = 1.0 / phase_mod_freq[0] / nu_laser

    dx = wavelength * focal_length / beam_aperture[0]
    dy = wavelength * focal_length / beam_aperture[1]
    Lx = dx * n_beamlets[0]
    Ly = dy * n_beamlets[1]
    lo = (0, 0)
    hi = (Lx, Ly)
    num_points = (160, 200)
    num_t = 400

    profile = SpeckleProfile(
        wavelength,
        polarization,
        focal_length,
        beam_aperture,
        n_beamlets,
        temporal_smoothing_type=temporal_smoothing_type,
        key=key,
        t_max=t_max,
        relative_laser_bandwidth=relative_laser_bandwidth,
        ssd_phase_modulation_amplitude=ssd_phase_modulation_amplitude,
        ssd_number_color_cycles=ssd_number_color_cycles,
        ssd_transverse_bandwidth_distribution=ssd_transverse_bandwidth_distribution,
    )

    x, y = create_2d_grid(lo, hi, num_points)
    t_array = jnp.linspace(0, t_max, num_t)
    F = evaluate_over_time(profile, x, y, t_array)

    period_error = jnp.abs(F[:, :, 0] - F[:, :, -1]).max() / jnp.abs(F).max()
    assert period_error < 1.0e-8


if __name__ == "__main__":
    # Run a quick smoke test
    print("Testing RPP intensity distribution...")
    test_intensity_distribution("RPP")
    print("PASSED")

    print("Testing FM SSD periodicity...")
    test_FM_SSD_periodicity()
    print("PASSED")

    print("All smoke tests passed!")
