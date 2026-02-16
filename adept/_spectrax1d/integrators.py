"""
Custom diffrax integrators for Spectrax-1D module.

This module contains specialized integrators that handle stiff damping terms
in the Spectrax-1D solver using operator splitting methods.

Integrators:
  - SplitStepDampingSolver: Wraps any RK solver with analytical boundary damping
  - LawsonRK4Solver: Exponential integrator using Lawson-RK4 method
"""

from typing import Optional

import equinox as eqx
import jax
from diffrax import RESULTS, AbstractSolver, AbstractTerm, Dopri8, LocalLinearInterpolation
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike

from adept._spectrax1d.exponential_operators import CombinedLinearExponential


class SplitStepDampingSolver(AbstractSolver):
    """
    Custom diffrax solver that combines any RK solver with analytical exponential damping.

    This split-step method solves stiffness issues in sponge boundary layers by treating
    the damping exactly rather than within the RK substeps.

    Mathematical approach:
        Standard ODE: dC/dt = F(C) - sigma(x)*C (stiff when sigma is large)   noqa: RUF003
        Split-step:
            Step 1: dC/dt = F(C)                      [RK integration]
            Step 2: C_new = C_old * exp(-sigma(x)*dt)     [Exact damping] # noqa: RUF003

    State structure (for spectrax-1d):
        y = {
            "Ck_electrons": Array (Np_e, Nm_e, Nn_e, Ny, Nx, Nz) as float64 view,
            "Ck_ions": Array (Np_i, Nm_i, Nn_i, Ny, Nx, Nz) as float64 view,
            "Fk": Array (6, Ny, Nx, Nz) as float64 view
        }

    Sponge profiles:
        - sponge_fields: Array (Nx,) - damping coefficient for EM fields
        - sponge_plasma_e: Array (Nx,) - damping coefficient for electrons
        - sponge_plasma_i: Array (Nx,) - damping coefficient for ions

    Args:
        wrapped_solver: The base diffrax solver (typically Dopri8, Tsit5, etc.)
        sponge_fields: Spatial damping profile for EM fields sigma_fields(x), shape (Nx,)
        sponge_plasma_e: Spatial damping profile for electron distribution sigma_e(x), shape (Nx,)
        sponge_plasma_i: Spatial damping profile for ion distribution sigma_i(x), shape (Nx,)
        Ny, Nx, Nz: Grid dimensions for proper broadcasting

    Benefits:
        - Unconditionally stable for any damping rate sigma
        - More accurate than implicit methods for large sigma
        - Allows 2-10x larger timesteps in absorbing regions
        - No modifications needed to physics equations
    """

    wrapped_solver: AbstractSolver
    sponge_fields: ArrayLike | None
    sponge_plasma_e: ArrayLike | None
    sponge_plasma_i: ArrayLike | None
    Ny: int
    Nx: int
    Nz: int

    def __init__(
        self,
        wrapped_solver: AbstractSolver = None,
        sponge_fields: ArrayLike | None = None,
        sponge_plasma_e: ArrayLike | None = None,
        sponge_plasma_i: ArrayLike | None = None,
        Ny: int = 1,
        Nx: int = 1,
        Nz: int = 1,
    ):
        """Initialize the split-step damping solver."""
        self.wrapped_solver = wrapped_solver if wrapped_solver is not None else Dopri8()
        self.sponge_fields = sponge_fields
        self.sponge_plasma_e = sponge_plasma_e
        self.sponge_plasma_i = sponge_plasma_i
        self.Ny = Ny
        self.Nx = Nx
        self.Nz = Nz

    @property
    def term_structure(self):
        """Delegate term_structure to wrapped solver."""
        return self.wrapped_solver.term_structure

    @property
    def interpolation_cls(self):
        """Delegate interpolation_cls to wrapped solver."""
        return self.wrapped_solver.interpolation_cls

    def order(self, terms):
        """Return the order of accuracy of the wrapped solver."""
        return self.wrapped_solver.order(terms)

    def init(self, terms, t0, t1, y0, args):
        """Initialize solver state by delegating to wrapped solver."""
        return self.wrapped_solver.init(terms, t0, t1, y0, args)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        """
        Perform one timestep with split-step method.

        Algorithm:
            1. Call wrapped solver (e.g., Dopri8) to integrate advection: dC/dt = F(C)
            2. Apply analytical exponential damping to the result

        Args:
            terms: ODETerm containing the vector field
            t0: Start time
            t1: End time
            y0: Initial state dictionary
            args: Runtime arguments
            solver_state: Solver-specific state
            made_jump: Whether a discontinuity was encountered

        Returns:
            (y1, error, dense_info, solver_state, result_flag)
        """
        # Step 1: Advection with wrapped solver (no damping in vector field)
        y1_advected, error, dense_info, solver_state, result = self.wrapped_solver.step(
            terms, t0, t1, y0, args, solver_state, made_jump
        )

        # Step 2: Apply exponential damping
        dt = t1 - t0
        y1_damped = self._apply_damping(y1_advected, dt)

        # Note: We keep the error estimate from the advection step
        # This is conservative and appropriate for operator splitting
        return y1_damped, error, dense_info, solver_state, result

    def _apply_damping(self, y: dict, dt: float) -> dict:
        """
        Apply exponential damping to state components.

        For each component with damping:
            C_new(x) = C_old(x) * exp(-sigma(x) * dt) # noqa: RUF003

        Implementation details:
        - Damping is applied in real space (sigma(x) is spatially varying)
        - sigma(x) is broadcast from (Nx,) to full array shapes # noqa: RUF003
        - Uses exp(-sigma*dt) multiplication (exact solution to dC/dt = -sigma*C) # noqa: RUF003

        Args:
            y: State dictionary with float64 views of complex arrays
            dt: Timestep size

        Returns:
            Damped state dictionary with float64 views
        """
        y_damped = {}

        # Apply damping to electromagnetic fields if configured
        if self.sponge_fields is not None:
            # Fk has shape (6, Ny, Nx, Nz) as float64 view of complex array
            Fk = y["Fk"].view(jnp.complex128)

            # Transform to real space, apply damping, transform back
            # This is necessary because sigma(x) is spatially varying
            E_real = jnp.fft.ifftn(Fk[:3], axes=(-3, -2, -1), norm="forward")
            B_real = jnp.fft.ifftn(Fk[3:], axes=(-3, -2, -1), norm="forward")

            # Broadcast sigma(x) from (Nx,) to (Ny, Nx, Nz)
            sigma_xyz = jnp.broadcast_to(self.sponge_fields[None, :, None], (self.Ny, self.Nx, self.Nz))

            # Apply exponential damping: C_new = C_old * exp(-sigma * dt)
            damping_factor = jnp.exp(-sigma_xyz * dt)
            E_damped = E_real * damping_factor[None, ...]  # Broadcast to (3, Ny, Nx, Nz)
            B_damped = B_real * damping_factor[None, ...]  # Broadcast to (3, Ny, Nx, Nz)

            # Transform back to k-space
            Ek_damped = jnp.fft.fftn(E_damped, axes=(-3, -2, -1), norm="forward")
            Bk_damped = jnp.fft.fftn(B_damped, axes=(-3, -2, -1), norm="forward")
            Fk_damped = jnp.concatenate([Ek_damped, Bk_damped], axis=0)

            y_damped["Fk"] = Fk_damped.view(jnp.float64)
        else:
            y_damped["Fk"] = y["Fk"]

        # Apply damping to electron distribution if configured
        if self.sponge_plasma_e is not None:
            # Ck_electrons has shape (Np_e, Nm_e, Nn_e, Ny, Nx, Nz) as float64 view
            Ck_e = y["Ck_electrons"].view(jnp.complex128)
            Np_e, Nm_e, Nn_e = Ck_e.shape[:3]

            # Transform to real space
            C_real_e = jnp.fft.ifftn(Ck_e, axes=(-3, -2, -1), norm="forward")

            # Broadcast sigma(x) from (Nx,) to (Ny, Nx, Nz)
            sigma_xyz = jnp.broadcast_to(self.sponge_plasma_e[None, :, None], (self.Ny, self.Nx, self.Nz))

            # Apply exponential damping ONLY to non-density Hermite modes
            # The (p=0, m=0, n=0) mode represents density and should not be damped
            # Create Hermite mode indices
            p_idx = jnp.arange(Np_e)[:, None, None, None, None, None]
            m_idx = jnp.arange(Nm_e)[None, :, None, None, None, None]
            n_idx = jnp.arange(Nn_e)[None, None, :, None, None, None]

            # Mask: 1 for density mode (no damping), exp(-sigma*dt) for wave modes
            is_density_mode = (p_idx == 0) & (m_idx == 0) & (n_idx == 0)
            spatial_damping = jnp.exp(-sigma_xyz * dt)[None, None, None, ...]  # (1,1,1,Ny,Nx,Nz)

            # Apply damping selectively: density mode undamped, wave modes damped
            damping_factor = jnp.where(is_density_mode, 1.0, spatial_damping)
            C_damped_e = C_real_e * damping_factor

            # Transform back to k-space
            Ck_damped_e = jnp.fft.fftn(C_damped_e, axes=(-3, -2, -1), norm="forward")
            y_damped["Ck_electrons"] = Ck_damped_e.view(jnp.float64)
        else:
            y_damped["Ck_electrons"] = y["Ck_electrons"]

        # Apply damping to ion distribution if configured
        if self.sponge_plasma_i is not None:
            # Ck_ions has shape (Np_i, Nm_i, Nn_i, Ny, Nx, Nz) as float64 view
            Ck_i = y["Ck_ions"].view(jnp.complex128)
            Np_i, Nm_i, Nn_i = Ck_i.shape[:3]

            # Transform to real space
            C_real_i = jnp.fft.ifftn(Ck_i, axes=(-3, -2, -1), norm="forward")

            # Broadcast sigma(x) from (Nx,) to (Ny, Nx, Nz)
            sigma_xyz = jnp.broadcast_to(self.sponge_plasma_i[None, :, None], (self.Ny, self.Nx, self.Nz))

            # Apply exponential damping ONLY to non-density Hermite modes
            # The (p=0, m=0, n=0) mode represents density and should not be damped
            # Create Hermite mode indices
            p_idx = jnp.arange(Np_i)[:, None, None, None, None, None]
            m_idx = jnp.arange(Nm_i)[None, :, None, None, None, None]
            n_idx = jnp.arange(Nn_i)[None, None, :, None, None, None]

            # Mask: 1 for density mode (no damping), exp(-sigma*dt) for wave modes
            is_density_mode = (p_idx == 0) & (m_idx == 0) & (n_idx == 0)
            spatial_damping = jnp.exp(-sigma_xyz * dt)[None, None, None, ...]  # (1,1,1,Ny,Nx,Nz)

            # Apply damping selectively: density mode undamped, wave modes damped
            damping_factor = jnp.where(is_density_mode, 1.0, spatial_damping)
            C_damped_i = C_real_i * damping_factor

            # Transform back to k-space
            Ck_damped_i = jnp.fft.fftn(C_damped_i, axes=(-3, -2, -1), norm="forward")
            y_damped["Ck_ions"] = Ck_damped_i.view(jnp.float64)
        else:
            y_damped["Ck_ions"] = y["Ck_ions"]

        return y_damped

    def func(self, terms, t0, y0, args):
        """
        Compute the vector field at a given point.

        This is used by adaptive stepsize controllers to evaluate the RHS.
        We delegate to the wrapped solver.
        """
        return self.wrapped_solver.func(terms, t0, y0, args)


# --- Pytree arithmetic helpers for state dictionaries ---


def _tree_add(a: dict, b: dict) -> dict:
    """Elementwise add two state dicts."""
    return jax.tree.map(lambda x, y: x + y, a, b)


def _tree_scale(a: dict, c: float) -> dict:
    """Scale all arrays in a state dict by a scalar."""
    return jax.tree.map(lambda x: c * x, a)


class LawsonRK4Solver(AbstractSolver):
    """
    Lawson-RK4 exponential integrator for the Vlasov-Maxwell system.

    Solves dy/dt = L·y + N(y,t) where L is handled by exact exponentials
    and N is advanced with classical RK4 in the integrating factor frame.

    Algorithm per step (dt = t1 - t0):

        Eₕ(·) = exp(L·dt/2),  Ef(·) = exp(L·dt)

        N₁ = N(yₙ, tₙ)
        y*  = Eₕ(yₙ) + (dt/2)·Eₕ(N₁)
        N₂ = N(y*, tₙ + dt/2)

        y** = Eₕ(yₙ) + (dt/2)·N₂
        N₃ = N(y**, tₙ + dt/2)

        y*** = Ef(yₙ) + dt·Eₕ(N₃)
        N₄ = N(y***, tₙ + dt)

        yₙ₊₁ = Ef(yₙ) + (dt/6)·[Ef(N₁) + 2·Eₕ(N₂) + 2·Eₕ(N₃) + N₄]

    This is 4th-order accurate and treats all linear stiffness exactly.
    Uses fixed timestep (no adaptive error estimate).

    Args:
        combined_exp: CombinedLinearExponential handling exp(L·s) for all state components
    """

    term_structure = AbstractTerm
    interpolation_cls = LocalLinearInterpolation

    combined_exp: CombinedLinearExponential

    def __init__(self, combined_exp: CombinedLinearExponential):
        self.combined_exp = combined_exp

    def order(self, terms):
        return 4

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        """
        Perform one Lawson-RK4 step.

        Args:
            terms: ODETerm wrapping the NonlinearVectorField
            t0, t1: Step interval
            y0: State dict (float64 views)
            args: Physical parameters
            solver_state: Unused (stateless)
            made_jump: Unused

        Returns:
            (y1, y_error, dense_info, solver_state, result)
        """
        dt = t1 - t0
        N = terms.vf  # NonlinearVectorField.__call__
        exp_L = self.combined_exp.apply  # exp_L(y, s) applies exp(L·s)

        # Cache repeated exponential applications
        Eh_yn = exp_L(y0, dt / 2)
        Ef_yn = exp_L(y0, dt)

        # --- Stage 1 ---
        N1 = N(t0, y0, args)

        # --- Stage 2 ---
        Eh_N1 = exp_L(N1, dt / 2)
        y_star = _tree_add(Eh_yn, _tree_scale(Eh_N1, dt / 2))
        N2 = N(t0 + dt / 2, y_star, args)

        # --- Stage 3 ---
        y_dstar = _tree_add(Eh_yn, _tree_scale(N2, dt / 2))
        N3 = N(t0 + dt / 2, y_dstar, args)

        # --- Stage 4 ---
        Eh_N3 = exp_L(N3, dt / 2)
        y_tstar = _tree_add(Ef_yn, _tree_scale(Eh_N3, dt))
        N4 = N(t1, y_tstar, args)

        # --- Combine ---
        Ef_N1 = exp_L(N1, dt)  # exp(L·dt) applied to N1
        Eh_N2 = exp_L(N2, dt / 2)
        Eh_N3_comb = Eh_N3  # Already computed above

        weighted = _tree_scale(
            _tree_add(
                _tree_add(Ef_N1, _tree_scale(Eh_N2, 2.0)),
                _tree_add(_tree_scale(Eh_N3_comb, 2.0), N4),
            ),
            dt / 6.0,
        )
        y1 = _tree_add(Ef_yn, weighted)

        # Dense info for interpolation (linear between endpoints)
        dense_info = dict(y0=y0, y1=y1)

        return y1, None, dense_info, solver_state, RESULTS.successful

    def func(self, terms, t0, y0, args):
        """Evaluate the nonlinear vector field (used by stepsize controllers)."""
        return terms.vf(t0, y0, args)
