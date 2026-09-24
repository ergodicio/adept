import numpy as np
from jax import Array, lax
from jax import numpy as jnp

from adept._lpse2d.core.stencils import check_order, first_derivative, injector_weights, second_derivative
from adept._lpse2d.core.vector import transverse_part  # re-exported for light.py and the tests


def light_absorption_rates(cfg: dict) -> tuple[float | None, float | None]:
    """Amplitude absorption rates (1/ps) at the critical density of the pump and of the
    Raman light (LPSE's per-class ``{laser|raman}.evolution.absorption``, ``wsAbsorptionAtNc``).
    ``terms.light.absorption`` is the pump's: ``false`` (none), ``true`` (NRL plasma formulary as
    coded in LPSE ``LightSolver.cpp``: ``nu = 5.11e10 Z logLambda / (lambda_um^2 Te_keV^1.5) *
    1e-12``, ``logLambda = 6.68 + ln(lambda_um Te)`` for ``Te > 0.01 Z^2`` else ``9.13 +
    ln(lambda_um Te^1.5 / Z)``, with the wave's own wavelength) or a number.
    ``terms.light.raman_absorption`` is the Raman light's, the same choices; when it is absent
    the Raman rate follows the pump's (the NRL formula at the Raman wavelength, or a number scaled
    by ``(lambda_1/lambda_0)^-2 = (w1/w0)^2``, the formula's leading dependence). The combined
    solver's field is the Raman class with carrier ``wp0``: its wavelength is ``lambda_0 w0/wp0``."""
    from astropy.units import Quantity as _Q

    light = cfg["terms"].get("light", {})
    derived = cfg["units"]["derived"]
    z = float(cfg["units"]["ionization state"])
    te = _Q(cfg["units"]["reference electron temperature"]).to("keV").value
    lambda0 = _Q(cfg["units"]["laser_wavelength"]).to("um").value
    combined = cfg["terms"]["epw"].get("solver", "separate") == "combined"
    lambda1 = lambda0 * derived["w0"] / (derived["wp0"] if combined else derived["w1"])

    def nrl(lam_um):
        if te > 0.01 * z**2:
            log_lambda = 6.68 + np.log(lam_um * te)
        else:
            log_lambda = 9.13 + np.log(lam_um * te**1.5 / z)
        return 5.11e10 * z * log_lambda / (lam_um**2 * te**1.5) * 1.0e-12

    def rate(value, lam_um):
        if value is None or value is False:
            return None
        if value is True:
            return float(nrl(lam_um))
        if float(value) < 0.0:
            raise ValueError("terms.light.absorption / raman_absorption must be false, true or a non-negative rate")
        return float(value)

    absorption = light.get("absorption", False)
    rate0 = rate(absorption, lambda0)
    raman = light.get("raman_absorption")
    if raman is not None:
        rate1 = rate(raman, lambda1)
    elif absorption is True:
        rate1 = float(nrl(lambda1))
    else:
        rate1 = None if rate0 is None else rate0 * (derived["w1"] / derived["w0"]) ** 2
    return rate0, rate1


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

    The spatial stencils are LPSE's ``evolution.solverOrder`` family (``terms.light.fd_order``
    2, 4 or 6; ``SchrodingerSolver3::step_2d``): central second differences of that order
    and, for the cross derivative, the product of the matching first-difference stencils.
    The default 2 is the MATLAB prototype's compact stencil. The plane-wave injectors are
    the total-field / scattered-field commutator of the same stencil with the injection
    mask (``stencils.injector_weights``), which at second order is the two-point source of
    the prototype and at higher order spreads over ``order / 2`` cells on either side of
    the plane, as LPSE's ``addPlanewaveSource`` does.

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
        # FD stencil order (terms.light.fd_order; LPSE evolution.solverOrder): (offset, weight)
        # pairs of the second- and first-difference stencils, the zero centre weight dropped
        self.fd_order = check_order(cfg["terms"].get("light", {}).get("fd_order", 2))
        m = self.fd_order // 2
        c2, c1 = second_derivative(self.fd_order), first_derivative(self.fd_order)
        self._second = [(j, float(c2[j + m])) for j in range(-m, m + 1)]
        self._first = [(j, float(c1[j + m])) for j in range(-m, m + 1) if c1[j + m] != 0.0]
        self.dt = cfg["grid"]["dt"]  # outer (EPW) step
        self.n_sub = cfg["grid"]["light_substeps"]
        self.dt_l = self.dt / self.n_sub  # light sub-step
        self.x = cfg["grid"]["x"]
        self.y = cfg["grid"]["y"]
        self.k_sq = cfg["grid"]["kx"][:, None] ** 2 + cfg["grid"]["ky"][None, :] ** 2
        self.kx_arr = jnp.asarray(cfg["grid"]["kx"])
        self.ky_arr = jnp.asarray(cfg["grid"]["ky"])
        self.one_over_k_sq = jnp.asarray(np.where(self.k_sq > 0, 1.0 / np.where(self.k_sq > 0, self.k_sq, 1.0), 0.0))
        # terms.light.transverse_source (LPSE takeTransversePartOfSourceTerms): the SRS
        # source is projected onto its transverse part. The longitudinal part of E1 is not
        # moved by any light propagator, so without the projection it accumulates the
        # source and pairs with the EPW in a spurious two-wave instability (in the test_006
        # cross-check: 41/ps energy growth against LPSE's 5.8/ps)
        self.transverse_source = bool(cfg["terms"].get("light", {}).get("transverse_source", True))
        # terms.light.transverse_fields: the FD curl-curl propagator (compact 3-point second
        # differences, centred cross difference) has a non-zero discrete divergence, so a
        # 2-D-structured transverse field acquires a longitudinal part at the discretisation
        # level every step (percent level at k0 dx ~ 1-2; an oblique plane wave at 5 deg: 5 %).
        # That part does not propagate and, driven by the EPW sources that see the whole
        # field, is exactly what the projected pump-depletion term cannot return energy from
        # (plan 2 N.4). Project the evolved light fields onto their transverse part once per
        # EPW step, as the spectral solver keeps them by construction. Off by default: on the
        # 1600x400 20 ps testbed case the projected FD scheme goes non-finite at 1.31 ps with
        # the EPW still at its noise floor (unprojected: 3.93 ps), so the projection is a
        # study option, not a fix
        self.transverse_fields = bool(cfg["terms"].get("light", {}).get("transverse_fields", False))

        background_density = cfg["grid"]["background_density"]
        # local detuning of the Raman envelope (MATLAB line 1668-1670)
        self.linear_coeff = (
            1j * self.w1 / 2.0 * (1.0 - self.wp0**2 / self.w1**2 * background_density / self.envelope_density)
        )
        self.diffraction_coeff = 1j * self.c**2 / (2.0 * self.w1)
        self.srs_coeff = -1j * self.e / (4.0 * self.w0 * self.me)

        # absorbing boundaries are applied every sub-step so that light (group velocity ~ c)
        # cannot cross the absorber between damping applications
        # the Raman light's layer (LPSE raman.evolution.Labc / abc.maxDampingRate); the pump's own
        # (laser.evolution.Labc) for an evolved pump (CoupledLight)
        grid = cfg["grid"]
        self.sub_boundary = grid.get("raman_absorbing_boundaries", grid["light_absorbing_boundaries"]) ** (
            1.0 / self.n_sub
        )
        self.sub_boundary0 = grid["light_absorbing_boundaries"] ** (1.0 / self.n_sub)
        # terms.light.absorber: pml (LPSE {laser|raman}.evolution.abc.type = pml; plan 2 L.2)
        # replaces the multiplicative layer by a complex coordinate stretch of the Laplacian,
        # v = 1 / (1 + e^{i pi / pml_denominator} delta^4) with delta the depth into the layer
        # (SchrodingerSolver3::abc_compute), on the compact stencil in the layers as LPSE
        self.pml = str(cfg["terms"].get("light", {}).get("absorber", "exp")) == "pml"
        if self.pml:
            self.sub_boundary = jnp.ones_like(self.sub_boundary)
            light_w = float(cfg["grid"].get("light_boundary_width_um", np.nan))
            raman_w = float(cfg["grid"].get("raman_boundary_width_um", np.nan))
            if np.isfinite(light_w) and np.isfinite(raman_w) and abs(light_w - raman_w) > 1e-12:
                raise NotImplementedError(
                    "terms.light.absorber: pml with different pump and Raman layer widths (one PML profile)"
                )
            denominator = float(cfg["terms"]["light"].get("pml_denominator", 5.0))
            s_abc = np.exp(1j * np.pi / denominator)
            from astropy.units import Quantity as _Q

            width = float(cfg["grid"].get("raman_boundary_width_um", _Q(cfg["grid"]["boundary_width"]).to("um").value))
            x = np.asarray(cfg["grid"]["x"], dtype=np.float64)
            y = np.asarray(cfg["grid"]["y"], dtype=np.float64)
            boundary = cfg["terms"]["epw"]["boundary"]
            depth_x = np.zeros(x.size)
            if str(boundary.get("x", "periodic")) == "absorbing":
                xmin, xmax = float(cfg["grid"]["xmin"]), float(cfg["grid"]["xmax"])
                depth_x = np.clip(np.maximum(xmin + width - x, x - (xmax - width)) / width, 0.0, 1.0)
            depth_y = np.zeros(y.size)
            if y.size > 1 and str(boundary.get("y", "periodic")) == "absorbing":
                ymin, ymax = float(cfg["grid"]["ymin"]), float(cfg["grid"]["ymax"])
                depth_y = np.clip(np.maximum(ymin + width - y, y - (ymax - width)) / width, 0.0, 1.0)
            delta = np.maximum(depth_x[:, None], depth_y[None, :])
            self.pml_v = jnp.asarray(1.0 / (1.0 + s_abc * delta**4))
            self.pml_inner = jnp.asarray(np.where(delta > 0.0, 0.0, 1.0))
            # LPSE leaves the edge nodes of an absorbing axis un-updated (E = 0 there): the
            # wall the attenuated wave reflects from on its way back through the layer. On this
            # periodic (roll) grid the same wall keeps the layer's remnant from wrapping around
            wall = np.ones((x.size, y.size))
            if str(boundary.get("x", "periodic")) == "absorbing":
                wall[0, :] = wall[-1, :] = 0.0
            if y.size > 1 and str(boundary.get("y", "periodic")) == "absorbing":
                wall[:, 0] = wall[:, -1] = 0.0
            self.sub_boundary = jnp.asarray(wall)
            self.sub_boundary0 = self.sub_boundary

        # collisional (inverse-bremsstrahlung) absorption, terms.light.absorption: the
        # amplitude decays at nu_abs (n/nc_w)^2 per wave, nc_w its own critical density
        # (LPSE calculateScatteringPotential; nu_abs from the NRL formula at nc_w with
        # the wave's wavelength, or a user rate in 1/ps at nc_w)
        self.n_over_env = background_density / self.envelope_density
        # iaw_density (the local fraction delta n / n_b) in units of n_env: n_b / n_env (LPSE)
        from adept._lpse2d.core.iaw import iaw_feedback_factor

        self.iaw_feedback = iaw_feedback_factor(cfg, "raman")
        self.iaw_feedback0 = iaw_feedback_factor(cfg, "pump")
        self.n_over_nc0 = background_density  # n / nc (w0)
        self.n_over_nc1 = background_density * (self.w0 / self.w1) ** 2  # n / nc (w1)
        self.absorption_rate0, self.absorption_rate1 = light_absorption_rates(cfg)
        self.absorption_factor1 = (
            None if self.absorption_rate1 is None else jnp.exp(-self.absorption_rate1 * self.dt_l * self.n_over_nc1**2)
        )

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
            self.k1_inject = float(np.sqrt(self.w1**2 - self.wpe_sq_i1) / self.c)
            self.seed_enabled = True
        else:
            self.seed_enabled = False
        # seed polarization (drivers.E1.polarization, LPSE raman.N.polarization): the injector
        # writes cos(psi) to the in-plane transverse component y and sin(psi) to z (plan 2 F.2)
        psi = float(cfg["drivers"].get("E1", {}).get("derived", {}).get("polarization", 0.0))
        self.seed_weights = (float(np.cos(psi)), float(np.sin(psi)))

    def add_seed(self, E1: Array, source: Array) -> Array:
        """Add a scalar seed source (nx, ny) with the seed polarization: cos(psi) to y, sin(psi) to z."""
        for c, w in zip((1, 2), self.seed_weights, strict=True):
            if w == 0.0:
                continue
            if c >= E1.shape[-1]:
                raise ValueError("an out-of-plane (s-polarised) seed needs three-component light fields")
            E1 = E1.at[..., c].add(w * source)
        return E1

    # the second-order branches keep the prototype's expressions verbatim so the default
    # stays bit-identical; the general branches are the same stencils for any order
    def _d2x(self, f: Array) -> Array:
        if self.fd_order == 2:
            return (jnp.roll(f, -1, axis=0) - 2.0 * f + jnp.roll(f, 1, axis=0)) / self.dx**2
        return sum(w * jnp.roll(f, -j, axis=0) for j, w in self._second) / self.dx**2

    def _d2y(self, f: Array) -> Array:
        if self.fd_order == 2:
            return (jnp.roll(f, -1, axis=1) - 2.0 * f + jnp.roll(f, 1, axis=1)) / self.dy**2
        return sum(w * jnp.roll(f, -j, axis=1) for j, w in self._second) / self.dy**2

    def _dxdy(self, f: Array) -> Array:
        if self.fd_order == 2:
            return (
                jnp.roll(f, (-1, -1), axis=(0, 1))
                - jnp.roll(f, (1, -1), axis=(0, 1))
                - jnp.roll(f, (-1, 1), axis=(0, 1))
                + jnp.roll(f, (1, 1), axis=(0, 1))
            ) / (4.0 * self.dx * self.dy)
        return sum(wi * wj * jnp.roll(f, (-i, -j), axis=(0, 1)) for i, wi in self._first for j, wj in self._first) / (
            self.dx * self.dy
        )

    def _laplacian_pml(self, f: Array) -> Array:
        """The complex-stretched compact Laplacian of the PML layers (LPSE ``step_2d`` PML
        branch): ``sum_nb v (v + v_nb)/2 (f_nb - f) / h^2``, with the stencil-order Laplacian
        in the interior (``v = 1``)."""
        v = self.pml_v
        lap = jnp.zeros_like(f)
        for axis, h in ((0, self.dx), (1, self.dy)):
            if f.shape[axis] == 1:
                continue
            for shift in (1, -1):
                v_nb = jnp.roll(v, shift, axis=axis)
                f_nb = jnp.roll(f, shift, axis=axis)
                lap = lap + v * 0.5 * (v + v_nb) * (f_nb - f) / h**2
        return jnp.where(self.pml_inner > 0.0, self._d2x(f) + self._d2y(f), lap)

    def curl_curl(self, E: Array) -> list[Array]:
        """``-(curl curl E)`` per component with the FD stencils: the discrete curl-curl on the
        in-plane components, the plain Laplacian on E_z (k_z = 0). With the PML the Laplacian
        part is the stretched one, the grad-div part the plain stencil (as LPSE)."""
        ex, ey = E[..., 0], E[..., 1]
        if self.pml:
            out = [
                self._laplacian_pml(ex) - (self._d2x(ex) + self._dxdy(ey)),
                self._laplacian_pml(ey) - (self._dxdy(ex) + self._d2y(ey)),
            ]
            if E.shape[-1] == 3:
                out.append(self._laplacian_pml(E[..., 2]))
            return out
        out = [self._d2y(ex) - self._dxdy(ey), self._d2x(ey) - self._dxdy(ex)]
        if E.shape[-1] == 3:
            ez = E[..., 2]
            out.append(self._d2x(ez) + self._d2y(ez))
        return out

    def _dx(self, f: Array) -> Array:
        """First difference along x of the stencil order (``stencils.first_derivative``)."""
        return sum(w * jnp.roll(f, -j, axis=0) for j, w in self._first) / self.dx

    def _dy(self, f: Array) -> Array:
        return sum(w * jnp.roll(f, -j, axis=1) for j, w in self._first) / self.dy

    def injector_rows(self, i_plane: int, direction: int, wave) -> list[tuple[int, Array]]:
        """The rows ``(index, values)`` of the plane-wave injector of this stencil order for the
        analytic wave ``wave(i) -> (ny,)`` (the source's amplitude, carrier and phases at row
        ``i``) filling the rows above (``direction = +1``) or below (``-1``) the plane at
        ``i_plane``: ``S_r = sum_j c_j [H(r + j) - H(r)] wave(r + j)`` (``stencils.injector_weights``).
        Second order: ``[(i_plane, +wave(i_plane + 1)), (i_plane + 1, -wave(i_plane))]`` for
        ``+1``, the two-point source of the prototype."""
        rows = []
        for r, pairs in injector_weights(self.fd_order, direction).items():
            value = None
            for j, w in pairs:
                term = w * wave(i_plane + r + j)
                value = term if value is None else value + term
            rows.append((i_plane + r, value))
        return rows

    def calc_seed_source(self, t: float, seed_args: dict) -> list[tuple[int, Array]]:
        """
        Rows of the seed injector (MATLAB lines 1757-1769 at second order; ``injector_rows``
        for the stencil order), launching the leftward wave into ``x <= x[i1]``.

        Returns ``(row index, values)`` pairs to be added to the E1 RHS.
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

        def wave(i):
            return 1j * amp * envelope_y * jnp.exp(-1j * k1 * self.x[i] - 1j * self.w1 * dw1 * t)

        return self.injector_rows(self.i1, -1, wave)

    def rhs(
        self,
        t: float,
        E1: Array,
        E0: Array,
        laplacian_phi: Array,
        seed_args: dict | None,
        iaw_density: Array | None = None,
        couple: bool = True,
    ) -> Array:
        linear_coeff = self.linear_coeff
        if iaw_density is not None:
            # MATLAB: i*w1/2 * [1 - wp0^2/w1^2 * (n_b/n_env + Nelf)] E1
            linear_coeff = linear_coeff - 1j * self.wp0**2 / (2.0 * self.w1) * iaw_density * self.iaw_feedback

        # paraxial propagation with cross-derivative terms (MATLAB lines 1663-1671): the
        # discrete curl-curl on the in-plane components; the out-of-plane component (k_z = 0)
        # sees the plain Laplacian, -(curl curl E)_z = laplacian(E_z)
        comps = [E1[..., i] for i in range(E1.shape[-1])]
        k_e1 = [self.diffraction_coeff * cc + linear_coeff * e for cc, e in zip(self.curl_curl(E1), comps, strict=True)]

        # SRS coupling to the EPW (MATLAB lines 1684-1689, potential formulation);
        # CoupledLight switches it off here when it integrates the exchange exactly
        if couple:
            source = (self.srs_coeff * jnp.conj(laplacian_phi))[..., None] * E0
            if self.transverse_source:
                source = transverse_part(source, self.kx_arr, self.ky_arr, self.one_over_k_sq)
            k_e1 = [k + source[..., i] for i, k in enumerate(k_e1)]

        if seed_args is not None:
            rows = self.calc_seed_source(t, seed_args)
            for c, w in zip((1, 2), self.seed_weights, strict=True):
                if w == 0.0:
                    continue
                if c >= len(k_e1):
                    raise ValueError("an out-of-plane (s-polarised) seed needs three-component light fields")
                for i, row in rows:
                    k_e1[c] = k_e1[c].at[i, :].add(w * row)

        return jnp.stack(k_e1, axis=-1)

    def __call__(
        self,
        t: float,
        E1: Array,
        E0_fn,
        phi_k: Array,
        seed_args: dict | None,
        iaw_density: Array | None = None,
    ) -> Array:
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
            k1 = self.rhs(t_i, E1, E0_fn(t_i), laplacian_phi, seed_args, iaw_density)
            E1 = E1 + self.dt_l * jnp.real(k1)
            # imaginary-part update with the RHS at t_i + dt/2 (MATLAB lines 1400-1421)
            k2 = self.rhs(
                t_i + self.dt_l / 2.0,
                E1,
                E0_fn(t_i + self.dt_l / 2.0),
                laplacian_phi,
                seed_args,
                iaw_density,
            )
            E1 = E1 + 1j * self.dt_l * jnp.imag(k2)
            # absorbing boundaries (MATLAB lines 977-983) and collisional absorption
            E1 = E1 * self.sub_boundary[..., None]
            if absorb is not None:
                E1 = E1 * absorb
            return E1

        absorb = None
        if self.absorption_rate1 is not None:
            n_over_nc = (
                self.n_over_nc1
                if iaw_density is None
                else self.n_over_nc1 * (1.0 + iaw_density * self.iaw_feedback / self.n_over_env)
            )
            absorb = jnp.exp(-self.absorption_rate1 * self.dt_l * n_over_nc**2)[..., None]
        E1 = lax.fori_loop(0, self.n_sub, substep, E1)
        if self.transverse_fields:
            E1 = transverse_part(E1, self.kx_arr, self.ky_arr, self.one_over_k_sq)
        return E1
