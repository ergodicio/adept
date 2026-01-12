#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

from jax import numpy as jnp

from adept._base_ import get_envelope


class Driver:
    def __init__(self, xax, driver_key="ex"):
        self.xax = xax
        self.driver_key = driver_key

    def get_this_pulse(self, this_pulse: dict, current_time: jnp.float64):
        kk = this_pulse["k0"]
        ww = this_pulse["w0"]
        dw = this_pulse["dw0"]
        t_L = this_pulse["t_center"] - this_pulse["t_width"] * 0.5
        t_R = this_pulse["t_center"] + this_pulse["t_width"] * 0.5
        t_wL = this_pulse["t_rise"]
        t_wR = this_pulse["t_rise"]
        x_L = this_pulse["x_center"] - this_pulse["x_width"] * 0.5
        x_R = this_pulse["x_center"] + this_pulse["x_width"] * 0.5
        x_wL = this_pulse["x_rise"]
        x_wR = this_pulse["x_rise"]
        envelope_t = get_envelope(t_wL, t_wR, t_L, t_R, current_time)
        envelope_x = get_envelope(x_wL, x_wR, x_L, x_R, self.xax)

        return (
            envelope_t * envelope_x * jnp.abs(kk) * this_pulse["a0"] * jnp.sin(kk * self.xax - (ww + dw) * current_time)
        )

    def __call__(self, t, args):
        total_de = jnp.zeros_like(self.xax)

        for _, pulse in args["drivers"][self.driver_key].items():
            total_de += self.get_this_pulse(pulse, t)

        return total_de


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

    def __call__(self, a: jnp.ndarray, aold: jnp.ndarray, djy_array: jnp.ndarray, electron_charge: jnp.ndarray):
        if self.c > 0:
            d2dx2 = (a[:-2] - 2.0 * a[1:-1] + a[2:]) / self.dx**2.0
            # padded_a = jnp.concatenate([a[-1:], a, a[:1]])
            # d2dx2 = (padded_a[:-2] - 2.0 * padded_a[1:-1] + padded_a[2:]) / self.dx**2.0
            anew = (
                2.0 * a[1:-1]
                - aold[1:-1]
                + self.dt**2.0 * (self.c_sq * d2dx2 - electron_charge * a[1:-1] + djy_array[1:-1])
            )
            # anew = 2.0 * a - aold + self.dt**2.0 * (self.c_sq * d2dx2 - electron_charge * a + djy_array)
            return {"a": self.apply_2nd_order_abc(aold, a, anew), "prev_a": a}
        else:
            return {"a": a, "prev_a": aold}


class SpectralPoissonSolver:
    """Spectral Poisson solver for electrostatic field.

    Solves Poisson equation: ∇²φ = -ρ, E = -∇φ
    where ρ = Σ_s q_s ∫f_s dv_s is the total charge density from all species.

    For quasineutral plasmas, the total charge should sum to zero at initialization.
    """

    def __init__(self, one_over_kx, species_grids, species_params):
        """Initialize the Poisson solver.

        Args:
            one_over_kx: 1/kx array for spectral solve (with 0 for k=0 mode)
            species_grids: dict mapping species_name -> {"dv": dv, "v": v, ...}
            species_params: dict mapping species_name -> {"charge": q, "mass": m, ...}
        """
        super().__init__()
        self.one_over_kx = one_over_kx
        self.species_grids = species_grids
        self.species_params = species_params

    def compute_charge_density(self, f_dict):
        """Compute total charge density from all species.

        ρ = Σ_s q_s ∫f_s dv_s = Σ_s q_s * n_s

        Args:
            f_dict: dict mapping species_name -> f[nx, nv] distribution

        Returns:
            Total charge density array[nx]
        """
        rho = jnp.zeros_like(list(f_dict.values())[0][:, 0])

        for species_name, f_s in f_dict.items():
            q_s = self.species_params[species_name]["charge"]
            dv_s = self.species_grids[species_name]["dv"]
            n_s = jnp.sum(f_s, axis=1) * dv_s
            rho = rho + q_s * n_s

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
        # E = -∇φ, ∇²φ = -ρ => in Fourier space: E_k = i*k*φ_k = i*k*(ρ_k/k²) = i*ρ_k/k
        # But we want E such that ∂E/∂x = ρ (Gauss's law), so E_k = -i*ρ_k/k
        # The sign convention here: positive charge creates outward E field
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
        j = jnp.zeros_like(list(f_dict.values())[0][:, 0])

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
        species_name = list(species_grids.keys())[0]
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
        f = list(f_dict.values())[0]

        prev_ek = jnp.fft.fft(prev_ex, axis=0)
        fk = jnp.fft.fft(f, axis=0)
        new_ek = (
            prev_ek
            + self.charge * self.one_over_ikx * jnp.sum(fk * (jnp.exp(-1j * self.kx * dt * self.vx) - 1), axis=1) * self.dv
        )

        return jnp.real(jnp.fft.ifft(new_ek))


class ElectricFieldSolver:
    """Wrapper for electrostatic field solvers.

    Combines the self-consistent electrostatic field from Poisson/Ampere solve
    with the ponderomotive force from laser fields.
    """

    def __init__(self, cfg):
        super().__init__()

        species_grids = cfg["grid"]["species_grids"]
        species_params = cfg["grid"]["species_params"]

        if cfg["terms"]["field"] == "poisson":
            self.es_field_solver = SpectralPoissonSolver(
                one_over_kx=cfg["grid"]["one_over_kx"],
                species_grids=species_grids,
                species_params=species_params,
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
                    kx=cfg["grid"]["kx"],
                    one_over_kx=cfg["grid"]["one_over_kx"],
                    species_grids=species_grids,
                    species_params=species_params,
                )
                self.hampere = True
            else:
                raise NotImplementedError(f"ampere + {cfg['terms']['time']} has not yet been implemented")
        else:
            raise NotImplementedError("Field Solver: <" + cfg["solver"]["field"] + "> has not yet been implemented")
        self.dx = cfg["grid"]["dx"]

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
