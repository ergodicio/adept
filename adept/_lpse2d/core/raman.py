import numpy as np
from jax import Array, lax
from jax import numpy as jnp


class RamanLight:
    """
    Evolves the Raman scattered-light envelope E1.

    This is a port of the MATLAB `raman.solver = 'fd'` branch of m201805_matlabLpse_v11.m
    (evalLaserFieldUpdate, lines 1656-1704, and lightSplitStep, lines 1377-1425).

    The envelope equation (per component, 2D with cross-derivative terms) is

        dE1/dt = i c^2/(2 w1) * (transverse Laplacian) E1
                 + i w1/2 * (1 - wp0^2/w1^2 * n/n_env) * E1
                 - i e/(4 w0 me) * conj(laplacian phi) * E0        (SRS coupling)
                 + seed injection source                            (optional)

    where w1 = w0 - wp0 is the Raman envelope frequency and laplacian phi is computed
    spectrally from the EPW potential (MATLAB line 1604).

    Time integration is the same staggered explicit scheme as MATLAB's lightSplitStep:
    the real part is updated with the RHS evaluated at t, then the imaginary part with
    the RHS evaluated at t + dt/2. Because this scheme is only conditionally stable
    (dt < ~dx^2 w1 / c^2), the update is sub-cycled `light_substeps` times inside each
    EPW step; the EPW potential is held fixed during the sub-steps, which matches
    MATLAB's `lightStepsPerEpwStep` behavior.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        derived = cfg["units"]["derived"]
        self.c = derived["c"]
        self.w0 = derived["w0"]
        self.w1 = derived["w1"]
        self.wp0 = derived["wp0"]
        self.e = derived["e"]
        self.me = derived["me"]
        self.envelope_density = cfg["units"]["envelope density"]

        self.dx = cfg["grid"]["dx"]
        self.dy = cfg["grid"]["dy"]
        self.dt = cfg["grid"]["dt"]  # outer (EPW) step
        self.n_sub = cfg["grid"]["light_substeps"]
        self.dt_l = self.dt / self.n_sub  # light sub-step
        self.x = cfg["grid"]["x"]
        self.y = cfg["grid"]["y"]
        self.k_sq = cfg["grid"]["kx"][:, None] ** 2 + cfg["grid"]["ky"][None, :] ** 2

        background_density = cfg["grid"]["background_density"]
        # local detuning of the Raman envelope (MATLAB line 1668-1670)
        self.linear_coeff = (
            1j * self.w1 / 2.0 * (1.0 - self.wp0**2 / self.w1**2 * background_density / self.envelope_density)
        )
        self.diffraction_coeff = 1j * self.c**2 / (2.0 * self.w1)
        self.srs_coeff = -1j * self.e / (4.0 * self.w0 * self.me)

        # absorbing boundaries are applied every sub-step so that light (group velocity ~ c)
        # cannot cross the absorber between damping applications
        self.sub_boundary = cfg["grid"]["absorbing_boundaries"] ** (1.0 / self.n_sub)

        # seed injection (MATLAB lines 1757-1769): a two-point antisymmetric source that
        # launches a leftward-propagating (-x) wave at x = xmax - offset
        if "E1" in cfg["drivers"]:
            seed = cfg["drivers"]["E1"]["derived"]
            x_inject = cfg["grid"]["xmax"] - seed["offset"]
            self.i1 = int(np.argmin(np.abs(np.array(self.x) - x_inject)))
            wpe_i1 = self.w0 * np.sqrt(background_density[self.i1, 0])
            self.wpe_sq_i1 = float(wpe_i1**2)
            permittivity1 = 1.0 - self.wpe_sq_i1 / self.w1**2
            if permittivity1 <= 0:
                raise ValueError(
                    f"The Raman seed injector at x = {float(self.x[self.i1]):.2f} um sits at density "
                    f"{float(background_density[self.i1, 0]):.3f} nc, above the w1 critical density "
                    f"{(self.w1 / self.w0) ** 2:.3f} nc where the seed is evanescent. Lower density.max, "
                    "or move the injector with drivers.E1.offset, or remove drivers.E1 to run noise-seeded."
                )
            self.source_prefactor = self.c**2 / (2.0 * self.w1) / permittivity1**0.25 / self.dx**2
            self.seed_enabled = True
        else:
            self.seed_enabled = False

    def _d2x(self, f: Array) -> Array:
        return (jnp.roll(f, -1, axis=0) - 2.0 * f + jnp.roll(f, 1, axis=0)) / self.dx**2

    def _d2y(self, f: Array) -> Array:
        return (jnp.roll(f, -1, axis=1) - 2.0 * f + jnp.roll(f, 1, axis=1)) / self.dy**2

    def _dxdy(self, f: Array) -> Array:
        return (
            jnp.roll(f, (-1, -1), axis=(0, 1))
            - jnp.roll(f, (1, -1), axis=(0, 1))
            - jnp.roll(f, (-1, 1), axis=(0, 1))
            + jnp.roll(f, (1, 1), axis=(0, 1))
        ) / (4.0 * self.dx * self.dy)

    def calc_seed_source(self, t: float, seed_args: dict) -> tuple[Array, Array]:
        """
        Amplitude and phases for the two-point seed injector (MATLAB lines 1757-1769).

        Returns the two rows to be added to the E1_y RHS at self.i1 and self.i1 + 1.
        """
        dw1 = seed_args["delta_omega"]
        turn_on = 1.0 - jnp.exp(-((t / seed_args["turn_on_time"]) ** 2))
        amp = self.source_prefactor * seed_args["amplitude"] * turn_on

        if seed_args["yw"] > 0:
            envelope_y = jnp.exp(-((self.y / (seed_args["yw"] / 2.0)) ** 4))
        else:
            envelope_y = jnp.ones_like(self.y)

        # local seed wavenumber (MATLAB line 867)
        k1 = self.w1 / self.c * jnp.sqrt((1.0 + dw1) ** 2 - self.wpe_sq_i1 / self.w1**2)

        row_i1 = -1j * amp * envelope_y * jnp.exp(-1j * k1 * self.x[self.i1 + 1] - 1j * self.w1 * dw1 * t)
        row_i1p1 = 1j * amp * envelope_y * jnp.exp(-1j * k1 * self.x[self.i1] - 1j * self.w1 * dw1 * t)
        return row_i1, row_i1p1

    def rhs(self, t: float, E1: Array, E0: Array, laplacian_phi: Array, seed_args: dict | None) -> Array:
        e1x, e1y = E1[..., 0], E1[..., 1]

        # paraxial propagation with cross-derivative terms (MATLAB lines 1663-1671)
        k_e1x = self.diffraction_coeff * (self._d2y(e1x) - self._dxdy(e1y)) + self.linear_coeff * e1x
        k_e1y = self.diffraction_coeff * (self._d2x(e1y) - self._dxdy(e1x)) + self.linear_coeff * e1y

        # SRS coupling to the EPW (MATLAB lines 1684-1689, potential formulation)
        k_e1x += self.srs_coeff * jnp.conj(laplacian_phi) * E0[..., 0]
        k_e1y += self.srs_coeff * jnp.conj(laplacian_phi) * E0[..., 1]

        if seed_args is not None:
            row_i1, row_i1p1 = self.calc_seed_source(t, seed_args)
            k_e1y = k_e1y.at[self.i1, :].add(row_i1)
            k_e1y = k_e1y.at[self.i1 + 1, :].add(row_i1p1)

        return jnp.stack([k_e1x, k_e1y], axis=-1)

    def __call__(self, t: float, E1: Array, E0_fn, phi_k: Array, seed_args: dict | None) -> Array:
        """
        Advance E1 over one EPW step (self.n_sub staggered light sub-steps).

        :param t: time at the start of the EPW step
        :param E1: Raman field, shape (nx, ny, 2), complex
        :param E0_fn: callable t -> pump field of shape (nx, ny, 2)
        :param phi_k: EPW potential in k-space, held fixed during the sub-steps
        :param seed_args: derived driver parameters for the seed, or None
        """
        seed_args = seed_args if self.seed_enabled else None
        laplacian_phi = jnp.fft.ifft2(-self.k_sq * phi_k)

        def substep(i, E1):
            t_i = t + i * self.dt_l
            # real-part update with the RHS at t_i (MATLAB lines 1380-1397)
            k1 = self.rhs(t_i, E1, E0_fn(t_i), laplacian_phi, seed_args)
            E1 = E1 + self.dt_l * jnp.real(k1)
            # imaginary-part update with the RHS at t_i + dt/2 (MATLAB lines 1400-1421)
            k2 = self.rhs(t_i + self.dt_l / 2.0, E1, E0_fn(t_i + self.dt_l / 2.0), laplacian_phi, seed_args)
            E1 = E1 + 1j * self.dt_l * jnp.imag(k2)
            # absorbing boundaries (MATLAB lines 977-983)
            E1 = E1 * self.sub_boundary[..., None]
            return E1

        return lax.fori_loop(0, self.n_sub, substep, E1)
