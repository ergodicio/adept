#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

from jax import numpy as jnp

from adept._base_ import get_envelope
from adept._vlasov1d.grid import Grid
from adept._vlasov1d.simulation import EMDriver


class LongitudinalElectricFieldDriver:
    """Computes normalized E_tilde = ω * a0 * sin(kx - ωt) from EM drivers."""

    def __init__(self, xax, drivers: list[EMDriver]):
        self.xax = xax
        self.drivers = drivers

    def _single_driver_field(self, driver: EMDriver, current_time):
        kk = driver.k0
        ww = driver.w0
        dw = driver.dw0
        factor = driver.envelope(self.xax, current_time)
        return factor * (ww + dw) * driver.a0 * jnp.sin(kk * self.xax - (ww + dw) * current_time)

    def __call__(self, t, args):
        total_de = jnp.zeros_like(self.xax)
        for pulse in self.drivers:
            total_de += self._single_driver_field(pulse, t)
        return total_de


class TransverseCurrentSourceDriver:
    """Computes source term for the transverse wave equation.

    For extended sources (default): S = -ω² * a0 * envelope(x,t) * sin(kx - ωt)
    For point sources: S = (F0/dx) * time_envelope(t) * δ_i0 * sin(ωt)

    For point sources, the amplitude scaling uses vacuum dispersion. For a 1D wave
    equation with point source S = F0 * delta(x - x0) * sin(wt), the Green's function
    gives outgoing waves with amplitude F0 / (2*k*c^2). To get amplitude a0, we need
    F0 = 2*k*c^2*a0. Using vacuum k = w/c, this gives F0 = 2*w*c*a0. In plasma
    (k_plasma < k_vac), the actual amplitude will be slightly larger than a0 by a
    factor k_vac / k_plasma.

    Note: point sources radiate equally in both directions. Place the source near a
    boundary with absorbing BCs to get a unidirectional wave.
    """

    def __init__(self, xax, drivers: list[EMDriver], c: float = 0.0):
        self.xax = xax
        self.drivers = drivers
        dx = float(xax[1] - xax[0])

        self.point_source_masks = []
        self.point_source_scales = []
        for driver in drivers:
            if driver.is_point_source:
                center = driver.envelope.space_envelope.center
                i0 = jnp.argmin(jnp.abs(xax - center))
                mask = jnp.zeros_like(xax).at[i0].set(1.0)
                w_total = driver.w0 + driver.dw0
                F0 = 2.0 * w_total * c * driver.a0
                self.point_source_masks.append(mask)
                self.point_source_scales.append(F0 / dx)
            else:
                self.point_source_masks.append(None)
                self.point_source_scales.append(None)

    def _single_driver_source(self, driver: EMDriver, mask, scale, current_time):
        ww = driver.w0
        dw = driver.dw0
        w_total = ww + dw
        if driver.is_point_source:
            time_env = driver.envelope.time_envelope(current_time)
            return scale * time_env * mask * jnp.sin(w_total * current_time)
        else:
            kk = driver.k0
            factor = driver.envelope(self.xax, current_time)
            return -factor * w_total**2 * driver.a0 * jnp.sin(kk * self.xax - w_total * current_time)

    def __call__(self, t, args):
        total = jnp.zeros_like(self.xax)
        for driver, mask, scale in zip(self.drivers, self.point_source_masks, self.point_source_scales):
            total += self._single_driver_source(driver, mask, scale, t)
        return total


class WaveSolver:
    def __init__(self, c: jnp.float64, dx: jnp.float64, dt: jnp.float64):
        super().__init__()
        self.dx = dx
        self.c = c
        self.c_sq = c**2.0
        c_over_dx = c / dx
        self.dt = dt
        self.const = c_over_dx * dt
        # abc_const = (const - 1.0) / (const + 1.0)
        self.one_over_const = 1.0 / dt / c_over_dx

    def apply_2nd_order_abc(self, aold, a, anew):
        """
        Second order absorbing boundary conditions

        :param aold:
        :param a:
        :param anew:
        :param dt:
        :return:
        """

        coeff = -1.0 / (self.one_over_const + 2.0 + self.const)

        # # 2nd order ABC
        a_left = (self.one_over_const - 2.0 + self.const) * (anew[1] + aold[0])
        a_left += 2.0 * (self.const - self.one_over_const) * (a[0] + a[2] - anew[0] - aold[1])
        a_left -= 4.0 * (self.one_over_const + self.const) * a[1]
        a_left *= coeff
        a_left -= aold[2]
        a_left = jnp.array([a_left])

        a_right = (self.one_over_const - 2.0 + self.const) * (anew[-2] + aold[-1])
        a_right += 2.0 * (self.const - self.one_over_const) * (a[-1] + a[-3] - anew[-1] - aold[-2])
        a_right -= 4.0 * (self.one_over_const + self.const) * a[-2]
        a_right *= coeff
        a_right -= aold[-3]
        a_right = jnp.array([a_right])

        # commenting out first order damping
        # a_left = jnp.array([a[1] + abc_const * (anew[0] - a[0])])
        # a_right = jnp.array([a[-2] + abc_const * (anew[-1] - a[-1])])

        return jnp.concatenate([a_left, anew, a_right])

    def __call__(self, a: jnp.ndarray, aold: jnp.ndarray, djy_array: jnp.ndarray, electron_density: jnp.ndarray):
        if self.c > 0:
            d2dx2 = (a[:-2] - 2.0 * a[1:-1] + a[2:]) / self.dx**2.0
            # padded_a = jnp.concatenate([a[-1:], a, a[:1]])
            # d2dx2 = (padded_a[:-2] - 2.0 * padded_a[1:-1] + padded_a[2:]) / self.dx**2.0
            anew = (
                2.0 * a[1:-1]
                - aold[1:-1]
                + self.dt**2.0 * (self.c_sq * d2dx2 - electron_density * a[1:-1] + djy_array[1:-1])
            )
            # anew = 2.0 * a - aold + self.dt**2.0 * (self.c_sq * d2dx2 - electron_charge * a + djy_array)
            return {"a": self.apply_2nd_order_abc(aold, a, anew), "prev_a": a}
        else:
            return {"a": a, "prev_a": aold}


class SpectralPoissonSolver:
    """Spectral Poisson solver for electrostatic field.

    Solves Poisson equation: div^2(phi) = -rho, E = -grad(phi)
    where rho = sum_s q_s * integral(f_s dv_s) is the total charge density from all species.

    For quasineutral plasmas, the total charge should sum to zero at initialization.
    """

    def __init__(self, one_over_kx, species_grids, species_params, static_charge_density=None):
        """Initialize the Poisson solver.

        Args:
            one_over_kx: 1/kx array for spectral solve (with 0 for k=0 mode)
            species_grids: dict mapping species_name -> {"dv": dv, "v": v, ...}
            species_params: dict mapping species_name -> {"charge": q, "mass": m, ...}
            static_charge_density: Optional static background charge density array[nx].
                For single-species electron sims, this is the ion background that
                enforces quasineutrality.
        """
        super().__init__()
        self.one_over_kx = one_over_kx
        self.species_grids = species_grids
        self.species_params = species_params
        self.static_charge_density = static_charge_density

    def compute_charge_density(self, f_dict):
        """Compute total charge density from all species plus static background.

        rho = sum_s q_s * integral(f_s dv_s) + static_charge_density

        Args:
            f_dict: dict mapping species_name -> f[nx, nv] distribution

        Returns:
            Total charge density array[nx]
        """
        rho = jnp.zeros_like(next(iter(f_dict.values()))[:, 0])

        for species_name, f_s in f_dict.items():
            q_s = self.species_params[species_name]["charge"]
            dv_s = self.species_grids[species_name]["dv"]
            n_s = jnp.sum(f_s, axis=1) * dv_s
            rho = rho + q_s * n_s

        if self.static_charge_density is not None:
            rho = rho + self.static_charge_density

        return rho

    def __call__(self, f_dict: dict, prev_ex: jnp.ndarray, dt: jnp.float64):
        """Solve Poisson equation for electric field.

        Args:
            f_dict: dict mapping species_name -> f[nx, nv] distribution
            prev_ex: previous electric field (unused for Poisson, kept for interface)
            dt: time step (unused for Poisson, kept for interface)

        Returns:
            Electric field E[nx] from Poisson solve
        """
        rho = self.compute_charge_density(f_dict)
        # Poisson equation: div^2(phi) = -rho => -k^2 phi_k = -rho_k => phi_k = rho_k / k^2
        # E = -grad(phi) => E_k = -ik phi_k = -ik rho_k / k^2 = -i rho_k / k
        return jnp.real(jnp.fft.ifft(-1j * self.one_over_kx * jnp.fft.fft(rho)))


class AmpereSolver:
    """Ampere solver using current density to evolve electric field.

    Solves: a single Euler step of ∂E/∂t = -j, where j = Σ_s (q_s/m_s) ∫v f_s dv
    is the total current density from all species.
    """

    def __init__(self, species_grids, species_params):
        """Initialize the Ampere solver.

        Args:
            species_grids: dict mapping species_name -> {"dv": dv, "v": v[nv], ...}
            species_params: dict mapping species_name -> {"charge": q, "mass": m, ...}
        """
        super().__init__()
        self.species_grids = species_grids
        self.species_params = species_params

    def compute_current_density(self, f_dict):
        """Compute total current density from all species.

        j = Σ_s q_s ∫v f_s dv_s

        Args:
            f_dict: dict mapping species_name -> f[nx, nv] distribution

        Returns:
            Total current density array[nx]
        """
        j = jnp.zeros_like(next(iter(f_dict.values()))[:, 0])

        for species_name, f_s in f_dict.items():
            q_s = self.species_params[species_name]["charge"]
            v_s = self.species_grids[species_name]["v"]
            dv_s = self.species_grids[species_name]["dv"]
            # Current from this species: q_s * ∫v f_s dv
            j_s = jnp.sum(v_s[None, :] * f_s, axis=1) * dv_s
            j = j + q_s * j_s

        return j

    def __call__(self, f_dict: dict, prev_ex: jnp.ndarray, dt: jnp.float64):
        """Evolve electric field using Ampere's law.

        Args:
            f_dict: dict mapping species_name -> f[nx, nv] distribution
            prev_ex: previous electric field E[nx]
            dt: time step

        Returns:
            Updated electric field E[nx]
        """
        j = self.compute_current_density(f_dict)
        return prev_ex - dt * j


class HampereSolver:
    """Hamiltonian Ampere solver using spectral integration in time.

    This solver uses the characteristic method to integrate Ampere's law
    exactly along particle trajectories in Fourier space.

    Note: Currently only supports single-species due to the complexity of
    the spectral integration with different velocity grids. For multi-species,
    use the standard AmpereSolver instead.
    """

    def __init__(self, kx, one_over_kx, species_grids, species_params):
        """Initialize the Hamiltonian Ampere solver.

        Args:
            kx: wavenumber array kx[nx]
            one_over_kx: 1/kx array (with 0 for k=0 mode)
            species_grids: dict mapping species_name -> {"dv": dv, "v": v[nv], ...}
            species_params: dict mapping species_name -> {"charge": q, "mass": m, ...}
        """
        self.kx = kx[:, None]
        self.one_over_ikx = one_over_kx / 1j
        self.species_grids = species_grids
        self.species_params = species_params

        # For now, validate that we have exactly one species (limitation documented above)
        if len(species_grids) > 1:
            raise NotImplementedError(
                "HampereSolver currently only supports single-species simulations. "
                "For multi-species, use 'ampere' or 'poisson' field solver instead."
            )

        # Cache the single species' grid parameters
        species_name = next(iter(species_grids.keys()))
        self.vx = species_grids[species_name]["v"][None, :]
        self.dv = species_grids[species_name]["dv"]
        self.charge = species_params[species_name]["charge"]

    def __call__(self, f_dict: dict, prev_ex: jnp.ndarray, dt: jnp.float64):
        """Evolve electric field using Hamiltonian Ampere method.

        Args:
            f_dict: dict mapping species_name -> f[nx, nv] distribution
            prev_ex: previous electric field E[nx]
            dt: time step

        Returns:
            Updated electric field E[nx]
        """
        # Extract the single species distribution
        f = next(iter(f_dict.values()))

        prev_ek = jnp.fft.fft(prev_ex, axis=0)
        fk = jnp.fft.fft(f, axis=0)
        new_ek = (
            prev_ek
            + self.charge
            * self.one_over_ikx
            * jnp.sum(fk * (jnp.exp(-1j * self.kx * dt * self.vx) - 1), axis=1)
            * self.dv
        )

        return jnp.real(jnp.fft.ifft(new_ek))


class ElectricFieldSolver:
    """Wrapper for electrostatic field solvers.

    Combines the self-consistent electrostatic field from Poisson/Ampere solve
    with the ponderomotive force from laser fields.
    """

    def __init__(self, cfg: dict, grid: Grid):
        super().__init__()

        species_grids = cfg["grid"]["species_grids"]
        species_params = cfg["grid"]["species_params"]
        static_charge_density = cfg["grid"].get("ion_charge")

        if cfg["terms"]["field"] == "poisson":
            self.es_field_solver = SpectralPoissonSolver(
                one_over_kx=grid.one_over_kx,
                species_grids=species_grids,
                species_params=species_params,
                static_charge_density=static_charge_density,
            )
            self.hampere = False
        elif cfg["terms"]["field"] == "ampere":
            if cfg["terms"]["time"] == "leapfrog":
                self.es_field_solver = AmpereSolver(
                    species_grids=species_grids,
                    species_params=species_params,
                )
                self.hampere = False
            else:
                raise NotImplementedError(f"ampere + {cfg['terms']['time']} has not yet been implemented")
        elif cfg["terms"]["field"] == "hampere":
            if cfg["terms"]["time"] == "leapfrog":
                self.es_field_solver = HampereSolver(
                    kx=grid.kx,
                    one_over_kx=grid.one_over_kx,
                    species_grids=species_grids,
                    species_params=species_params,
                )
                self.hampere = True
            else:
                raise NotImplementedError(f"ampere + {cfg['terms']['time']} has not yet been implemented")
        else:
            raise NotImplementedError("Field Solver: <" + cfg["solver"]["field"] + "> has not yet been implemented")
        self.dx = grid.dx

    def __call__(self, f_dict: dict, a: jnp.ndarray, prev_ex: jnp.ndarray, dt: jnp.float64):
        """Compute total electrostatic field for the Vlasov equation.

        The total field is a sum of:
        - Ponderomotive force from E_y (laser field)
        - Self-consistent electrostatic field from Poisson or Ampere solve

        Args:
            f_dict: dict mapping species_name -> f[nx, nv] distribution
            a: vector potential a[nx+2] (with boundary cells)
            prev_ex: previous electric field E[nx]
            dt: time step

        Returns:
            Tuple of (ponderomotive_force[nx], self_consistent_ex[nx])
        """
        ponderomotive_force = -0.5 * jnp.gradient(a**2.0, self.dx)[1:-1]
        self_consistent_ex = self.es_field_solver(f_dict, prev_ex, dt)
        return ponderomotive_force, self_consistent_ex
