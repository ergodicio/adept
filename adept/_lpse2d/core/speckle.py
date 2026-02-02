"""
JAX port of LASY SpeckleProfile for generating speckled laser profiles.

Ported from: https://github.com/LASY-org/lasy/blob/development/lasy/profiles/speckle_profile.py

Speckled lasers are used to mitigate laser-plasma interactions in fusion and ion acceleration contexts.
More on the subject can be found in chapter 9 of P. Michel, Introduction to Laser-Plasma Interactions.
"""

import jax
import jax.numpy as jnp
from jax import Array
from scipy.constants import c


def gen_gaussian_time_series(
    key: Array,
    t_num: int,
    dt: float,
    fwhm: float,
    rms_mean: float,
) -> Array:
    """Generate a discrete time series that has gaussian power spectrum.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for JAX random number generation
    t_num : int
        The number of grid points in time
    dt : float
        Time step
    fwhm : float
        The full width half maximum of the power spectrum
    rms_mean : float
        The root-mean-square average of the spectrum

    Returns
    -------
    temporal_amplitude : 1darray
        A time series array of complex numbers with shape [t_num]
    """
    if fwhm == 0.0:
        return jnp.zeros(t_num, dtype=jnp.complex128)

    omega = jnp.fft.fftshift(jnp.fft.fftfreq(t_num, d=dt))
    psd = jnp.exp(-jnp.log(2) * 0.5 * jnp.square(omega / fwhm * 2 * jnp.pi))

    key1, key2 = jax.random.split(key)
    real_part = jax.random.normal(key1, shape=(t_num,))
    imag_part = jax.random.normal(key2, shape=(t_num,))
    spectral_amplitude = psd * (real_part + 1j * imag_part)

    temporal_amplitude = jnp.fft.ifftshift(jnp.fft.fft(jnp.fft.fftshift(spectral_amplitude)))
    temporal_amplitude = temporal_amplitude * rms_mean / jnp.sqrt(jnp.mean(jnp.square(jnp.abs(temporal_amplitude))))
    return temporal_amplitude


class SpeckleProfile:
    """
    JAX-based class for the profile of a speckled laser pulse.

    This is a JAX port of the LASY SpeckleProfile class. Key differences:
    - Uses JAX arrays and operations
    - Random state is initialized in constructor (key parameter)
    - evaluate() operates on a single time slice

    Supported smoothing types:
    - 'RPP': Random phase plates (static)
    - 'CPP': Continuous phase plates (static)
    - 'FM SSD': Frequency modulated smoothing by spectral dispersion (time-varying)
    - 'GP RPM SSD': Gaussian process randomly phase-modulated SSD (time-varying)
    - 'GP ISI': Gaussian process induced spatial incoherence (time-varying)

    Parameters
    ----------
    wavelength : float (in meters)
        The main laser wavelength of the laser.
    pol : list of 2 complex numbers (dimensionless)
        Polarization vector.
    focal_length : float (in meters)
        Focal length of lens just after the RPP/CPP.
    beam_aperture : list of 2 floats (in meters)
        Widths of the rectangular beam in the near-field.
    n_beamlets : list of 2 integers
        Number of RPP/CPP elements in each direction.
    temporal_smoothing_type : string
        Smoothing method: 'RPP', 'CPP', 'FM SSD', 'GP RPM SSD', or 'GP ISI'.
    key : jax.random.PRNGKey
        Random key for initializing phase plate, dephasing, and time series.
    t_max : float, optional (in seconds)
        Maximum simulation time. Required for GP methods.
    relative_laser_bandwidth : float, optional
        Bandwidth of the laser pulse, relative to central frequency.
        Required for SSD/ISI methods.
    ssd_phase_modulation_amplitude : list of 2 floats, optional
        Amplitudes of phase modulation (for SSD types).
    ssd_number_color_cycles : list of 2 floats, optional
        Number of color cycles (for SSD types).
    ssd_transverse_bandwidth_distribution : list of 2 floats, optional
        SSD bandwidth distribution (for SSD types).
    do_include_transverse_envelope : bool, optional
        Whether to include the transverse sinc envelope.
    """

    supported_smoothing = ("RPP", "CPP", "FM SSD", "GP RPM SSD", "GP ISI")

    def __init__(
        self,
        wavelength: float,
        pol: tuple,
        focal_length: float,
        beam_aperture: tuple,
        n_beamlets: tuple,
        temporal_smoothing_type: str,
        key: Array,
        t_max: float = 0.0,
        relative_laser_bandwidth: float = 1e-10,
        ssd_phase_modulation_amplitude: tuple | None = None,
        ssd_number_color_cycles: tuple | None = None,
        ssd_transverse_bandwidth_distribution: tuple | None = None,
        do_include_transverse_envelope: bool = False,
    ):
        # Base profile attributes
        norm_pol = jnp.sqrt(jnp.abs(pol[0]) ** 2 + jnp.abs(pol[1]) ** 2)
        self.pol = jnp.array([pol[0] / norm_pol, pol[1] / norm_pol])
        self.lambda0 = wavelength
        self.omega0 = 2 * jnp.pi * c / self.lambda0
        self.k0 = 2.0 * jnp.pi / wavelength

        # Speckle-specific attributes
        self.focal_length = focal_length
        self.beam_aperture = jnp.array(beam_aperture, dtype=jnp.float64)
        self.n_beamlets = tuple(n_beamlets)
        self.temporal_smoothing_type = temporal_smoothing_type.upper()
        self.laser_bandwidth = relative_laser_bandwidth
        self.do_include_transverse_envelope = do_include_transverse_envelope

        # Time interval to update the speckle pattern (for time-varying methods)
        self.dt_update = 1 / self.laser_bandwidth / 50

        # Beamlet grid in lens plane
        self.x_lens_list = jnp.linspace(
            -0.5 * (self.n_beamlets[0] - 1),
            0.5 * (self.n_beamlets[0] - 1),
            num=self.n_beamlets[0],
        )
        self.y_lens_list = jnp.linspace(
            -0.5 * (self.n_beamlets[1] - 1),
            0.5 * (self.n_beamlets[1] - 1),
            num=self.n_beamlets[1],
        )
        self.Y_lens_matrix, self.X_lens_matrix = jnp.meshgrid(self.y_lens_list, self.x_lens_list)
        self.Y_lens_index_matrix, self.X_lens_index_matrix = jnp.meshgrid(
            jnp.arange(self.n_beamlets[1], dtype=jnp.float64),
            jnp.arange(self.n_beamlets[0], dtype=jnp.float64),
        )

        # SSD parameters
        self.ssd_phase_modulation_amplitude = ssd_phase_modulation_amplitude
        self.ssd_number_color_cycles = ssd_number_color_cycles
        self.ssd_transverse_bandwidth_distribution = ssd_transverse_bandwidth_distribution

        if "SSD" in self.temporal_smoothing_type:
            ssd_normalization = jnp.sqrt(
                self.ssd_transverse_bandwidth_distribution[0] ** 2 + self.ssd_transverse_bandwidth_distribution[1] ** 2
            )
            ssd_frac = [
                self.ssd_transverse_bandwidth_distribution[0] / ssd_normalization,
                self.ssd_transverse_bandwidth_distribution[1] / ssd_normalization,
            ]
            self.ssd_phase_modulation_frequency = [
                self.laser_bandwidth * sf * 0.5 / pma
                for sf, pma in zip(ssd_frac, self.ssd_phase_modulation_amplitude, strict=True)
            ]
            self.ssd_time_delay = (
                (
                    self.ssd_number_color_cycles[0] / self.ssd_phase_modulation_frequency[0]
                    if self.ssd_phase_modulation_frequency[0] > 0
                    else 0
                ),
                (
                    self.ssd_number_color_cycles[1] / self.ssd_phase_modulation_frequency[1]
                    if self.ssd_phase_modulation_frequency[1] > 0
                    else 0
                ),
            )

        # Validate inputs
        assert self.temporal_smoothing_type in SpeckleProfile.supported_smoothing, (
            f"Only support one of: {', '.join(SpeckleProfile.supported_smoothing)}"
        )
        assert relative_laser_bandwidth > 0, "laser_bandwidth must be greater than 0"
        assert len(n_beamlets) == 2, "n_beamlets must be size 2"

        if "SSD" in self.temporal_smoothing_type:
            assert ssd_number_color_cycles is not None, "must supply `ssd_number_color_cycles` to use SSD"
            assert ssd_transverse_bandwidth_distribution is not None, (
                "must supply `ssd_transverse_bandwidth_distribution` to use SSD"
            )
            assert ssd_phase_modulation_amplitude is not None, "must supply `ssd_phase_modulation_amplitude` to use SSD"

        if "GP" in self.temporal_smoothing_type:
            assert t_max > 0, "t_max must be provided and > 0 for GP methods"

        # Pre-compute random state (split key for different random operations)
        key1, key2, key3 = jax.random.split(key, 3)

        # Phase plate (computed once, used for all evaluations)
        if self.temporal_smoothing_type == "RPP":
            phase_plate = jax.random.choice(key1, jnp.array([0.0, jnp.pi]), shape=self.n_beamlets)
        elif self.temporal_smoothing_type in ["CPP", "FM SSD", "GP RPM SSD"]:
            phase_plate = jax.random.uniform(key1, shape=self.n_beamlets, minval=-jnp.pi, maxval=jnp.pi)
        elif "ISI" in self.temporal_smoothing_type:
            phase_plate = jnp.zeros(self.n_beamlets)
        else:
            raise NotImplementedError(f"Unknown smoothing type: {self.temporal_smoothing_type}")

        self.exp_phase_plate = jnp.exp(1j * phase_plate)

        # SSD dephasing (for FM SSD only)
        self.ssd_x_y_dephasing = None
        if self.temporal_smoothing_type == "FM SSD":
            self.ssd_x_y_dephasing = jax.random.normal(key2, shape=(2,)) * jnp.pi

        # Time series for GP methods
        self.series_time = None
        self.time_series = None
        if "GP" in self.temporal_smoothing_type:
            t_max_norm = t_max * c / self.lambda0
            series_time = jnp.arange(0, t_max_norm + self.dt_update, self.dt_update)
            self.series_time, self.time_series = self._init_gaussian_time_series(key3, series_time)

    def _init_gaussian_time_series(
        self,
        key: Array,
        series_time: Array,
    ) -> tuple[Array, Array | None]:
        """Initialize a time series sampled from a Gaussian process.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key for JAX random number generation
        series_time : 1d array
            Array of times at which to sample from Gaussian process

        Returns
        -------
        time_interp : array
            Time array (possibly padded for SSD)
        time_series : array or None
            Random phase/amplitude data, or None for non-GP methods
        """
        if "SSD" in self.temporal_smoothing_type:
            ssd_time_delay_sum = self.ssd_time_delay[0] + self.ssd_time_delay[1]
            n_pts = series_time.size + int(ssd_time_delay_sum / self.dt_update) + 2

            key1, key2 = jax.random.split(key)
            pm_phase0 = gen_gaussian_time_series(
                key1,
                n_pts,
                self.dt_update,
                2 * jnp.pi * self.ssd_phase_modulation_frequency[0],
                self.ssd_phase_modulation_amplitude[0],
            )
            pm_phase1 = gen_gaussian_time_series(
                key2,
                n_pts,
                self.dt_update,
                2 * jnp.pi * self.ssd_phase_modulation_frequency[1],
                self.ssd_phase_modulation_amplitude[1],
            )
            time_interp = jnp.arange(
                0,
                series_time[-1] + ssd_time_delay_sum + 3 * self.dt_update,
                self.dt_update,
            )[: pm_phase0.size]
            return (
                time_interp,
                jnp.stack(
                    [
                        (jnp.real(pm_phase0) + jnp.imag(pm_phase0)) / jnp.sqrt(2),
                        (jnp.real(pm_phase1) + jnp.imag(pm_phase1)) / jnp.sqrt(2),
                    ]
                ),
            )
        elif "ISI" in self.temporal_smoothing_type:
            # Generate complex amplitudes for each beamlet
            keys = jax.random.split(key, self.n_beamlets[0] * self.n_beamlets[1])
            keys = keys.reshape(self.n_beamlets[0], self.n_beamlets[1], 2)

            def gen_beamlet_series(k):
                return gen_gaussian_time_series(k, series_time.size, self.dt_update, 2 * self.laser_bandwidth, 1)

            complex_amp = jax.vmap(jax.vmap(gen_beamlet_series))(keys)
            return series_time, complex_amp
        else:
            return series_time, None

    def beamlets_complex_amplitude(
        self,
        t_now: float,
    ) -> Array:
        """Calculate complex amplitude of the beamlets in the near-field.

        Parameters
        ----------
        t_now : float
            Time at which to calculate the complex amplitude (normalized by c/lambda0)

        Returns
        -------
        array of complex numbers giving beamlet amplitude and phases
        """
        if self.temporal_smoothing_type in ["RPP", "CPP"]:
            return jnp.ones_like(self.X_lens_matrix)

        elif self.temporal_smoothing_type == "FM SSD":
            phase_t = self.ssd_phase_modulation_amplitude[0] * jnp.sin(
                self.ssd_x_y_dephasing[0]
                + 2
                * jnp.pi
                * self.ssd_phase_modulation_frequency[0]
                * (t_now - self.X_lens_matrix * self.ssd_time_delay[0] / self.n_beamlets[0])
            ) + self.ssd_phase_modulation_amplitude[1] * jnp.sin(
                self.ssd_x_y_dephasing[1]
                + 2
                * jnp.pi
                * self.ssd_phase_modulation_frequency[1]
                * (t_now - self.Y_lens_matrix * self.ssd_time_delay[1] / self.n_beamlets[1])
            )
            return jnp.exp(1j * phase_t)

        elif self.temporal_smoothing_type == "GP RPM SSD":
            phase_t = jnp.interp(
                t_now + self.X_lens_index_matrix * self.ssd_time_delay[0] / self.n_beamlets[0],
                self.series_time,
                self.time_series[0],
            ) + jnp.interp(
                t_now + self.Y_lens_index_matrix * self.ssd_time_delay[1] / self.n_beamlets[1],
                self.series_time,
                self.time_series[1],
            )
            return jnp.exp(1j * phase_t)

        elif self.temporal_smoothing_type == "GP ISI":
            idx = jnp.round(t_now / self.dt_update).astype(int)
            return self.time_series[:, :, idx]

        else:
            raise NotImplementedError(f"Unknown smoothing type: {self.temporal_smoothing_type}")

    def generate_speckle_pattern(
        self,
        t_now: float,
        x: Array,
        y: Array,
    ) -> Array:
        """Calculate the speckle pattern in the focal plane.

        Parameters
        ----------
        t_now : float
            Time at which to calculate the speckle pattern (normalized by c/lambda0)
        x : 2d array
            x-positions in focal plane (meters)
        y : 2d array
            y-positions in focal plane (meters)

        Returns
        -------
        speckle_amp : 2D array of complex numbers
        """
        lambda_fnum = self.lambda0 * self.focal_length / self.beam_aperture
        X_focus_matrix = x / lambda_fnum[0]
        Y_focus_matrix = y / lambda_fnum[1]
        x_focus_list = X_focus_matrix[:, 0]
        y_focus_list = Y_focus_matrix[0, :]

        x_phase_focus_matrix = jnp.exp(
            -2 * jnp.pi * 1j / self.n_beamlets[0] * self.x_lens_list[:, jnp.newaxis] * x_focus_list[jnp.newaxis, :]
        )
        y_phase_focus_matrix = jnp.exp(
            -2 * jnp.pi * 1j / self.n_beamlets[1] * self.y_lens_list[:, jnp.newaxis] * y_focus_list[jnp.newaxis, :]
        )

        bca = self.beamlets_complex_amplitude(t_now)
        speckle_amp = jnp.einsum(
            "jk,jl->kl",
            jnp.einsum("ij,ik->jk", bca * self.exp_phase_plate, x_phase_focus_matrix),
            y_phase_focus_matrix,
        )
        if self.do_include_transverse_envelope:
            speckle_amp = (
                jnp.sinc(X_focus_matrix / self.n_beamlets[0])
                * jnp.sinc(Y_focus_matrix / self.n_beamlets[1])
                * speckle_amp
            )
        return speckle_amp

    def evaluate(self, x: Array, y: Array, t: float) -> Array:
        """
        Return the envelope field of the laser at a single time.

        Parameters
        ----------
        x, y : 2D ndarrays of floats (in meters)
            Define spatial points on which to evaluate the envelope.
            These arrays must have the same shape.
        t : float (in seconds)
            Time at which to evaluate the envelope.

        Returns
        -------
        envelope : 2D ndarray of complex numbers
            Contains the value of the envelope at the specified points.
            This array has the same shape as the arrays x, y.
        """
        t_norm = t * c / self.lambda0
        return self.generate_speckle_pattern(t_norm, x, y)
