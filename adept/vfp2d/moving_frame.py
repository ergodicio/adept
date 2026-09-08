"""Ion-frame kinematic operators for the spherical-harmonic electron state.

The mixed coordinates keep position and time in the laboratory frame while
representing velocity as ``c = v - u_i(x, t)``.  For
``A_ij = partial_j u_i`` and ``a_i = D_t u_i``, the frame terms are

``-div_x(u_i f) + div_c(A c f) + a_i . grad_c(f)``.

Together with relative streaming, these terms are the non-relativistic limit
of the mixed-frame VFP equation. The deformation operator compiles the angular
Galerkin projection into sparse real-harmonic coupling rules and retains the
dense quadrature implementation as a verification oracle.
"""

from __future__ import annotations

from math import factorial
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.special import lpmv

from adept.vfp2d.harmonics import HarmonicLayout, TzoufrasVlasov


class _AngularGalerkin:
    """Quadrature transforms for ADEPT's unnormalized associated-Legendre basis."""

    def __init__(self, layout: HarmonicLayout):
        n_mu = max(3, layout.l_max + 3)
        n_phi = max(5, 2 * layout.l_max + 5)
        mu, weight_mu = np.polynomial.legendre.leggauss(n_mu)
        phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
        sin_theta = np.sqrt(1.0 - mu**2)

        mu_grid, phi_grid = np.meshgrid(mu, phi, indexing="ij")
        sin_grid = np.sqrt(1.0 - mu_grid**2)
        directions = np.stack(
            (
                mu_grid,
                sin_grid * np.cos(phi_grid),
                sin_grid * np.sin(phi_grid),
            ),
            axis=-1,
        ).reshape(-1, 3)
        theta_directions = np.stack(
            (
                -sin_grid,
                mu_grid * np.cos(phi_grid),
                mu_grid * np.sin(phi_grid),
            ),
            axis=-1,
        ).reshape(-1, 3)
        phi_directions = np.stack(
            (
                np.zeros_like(phi_grid),
                -np.sin(phi_grid),
                np.cos(phi_grid),
            ),
            axis=-1,
        ).reshape(-1, 3)
        weights = (weight_mu[:, None] * np.full((1, n_phi), 2.0 * np.pi / n_phi)).reshape(-1)

        basis = []
        dtheta_basis = []
        dphi_basis = []
        norms = []
        multiplicity = []
        for ell, m in layout.pairs:
            # scipy includes the Condon--Shortley (-1)^m phase; ADEPT/Tzoufras does not.
            associated = (-1) ** m * lpmv(m, ell, mu)
            if ell == 0:
                derivative_mu = np.zeros_like(mu)
            else:
                lower = np.zeros_like(mu) if m > ell - 1 else (-1) ** m * lpmv(m, ell - 1, mu)
                derivative_mu = (ell * mu * associated - (ell + m) * lower) / (mu**2 - 1.0)
            phase = np.exp(1j * m * phi)
            harmonic = (associated[:, None] * phase[None, :]).reshape(-1)
            dtheta = (-sin_theta[:, None] * derivative_mu[:, None] * phase[None, :]).reshape(-1)
            basis.append(harmonic)
            dtheta_basis.append(dtheta)
            dphi_basis.append(1j * m * harmonic)
            norms.append(4.0 * np.pi * factorial(ell + m) / ((2 * ell + 1) * factorial(ell - m)))
            multiplicity.append(1.0 if m == 0 else 2.0)

        basis_array = np.asarray(basis)
        self.basis = jnp.asarray(basis_array)
        self.reconstruction_basis = jnp.asarray(np.asarray(multiplicity)[:, None] * basis_array)
        self.dtheta_reconstruction_basis = jnp.asarray(np.asarray(multiplicity)[:, None] * np.asarray(dtheta_basis))
        self.dphi_reconstruction_basis = jnp.asarray(np.asarray(multiplicity)[:, None] * np.asarray(dphi_basis))
        self.projection_basis = jnp.asarray(np.conjugate(basis_array) * weights[None, :] / np.asarray(norms)[:, None])
        self.coefficient_scale = jnp.asarray(np.sqrt(norms))
        self.directions = jnp.asarray(directions)
        self.theta_directions = jnp.asarray(theta_directions)
        self.phi_directions = jnp.asarray(phi_directions)
        self.sin_theta = jnp.asarray(np.repeat(sin_theta, n_phi))
        self.layout = layout

    @staticmethod
    def _evaluate(coefficients: Array, basis: Array) -> Array:
        return jnp.real(jnp.einsum("...hv,hq->...qv", coefficients, basis))

    def reconstruct(self, coefficients: Array) -> Array:
        return self._evaluate(coefficients, self.reconstruction_basis)

    def angular_derivatives(self, coefficients: Array) -> tuple[Array, Array]:
        return (
            self._evaluate(coefficients, self.dtheta_reconstruction_basis),
            self._evaluate(coefficients, self.dphi_reconstruction_basis),
        )

    def project(self, values: Array) -> Array:
        coefficients = jnp.einsum("...qv,hq->...hv", values, self.projection_basis)
        for index, (_ell, m) in enumerate(self.layout.pairs):
            if m == 0:
                coefficients = coefficients.at[..., index, :].set(jnp.real(coefficients[..., index, :]))
        return coefficients


class _SparseAngularCoupling:
    """Sparse real-linear harmonic maps compiled from the Galerkin oracle."""

    _operator_cache: ClassVar[dict[tuple[int, int], tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]]] = {}

    def __init__(self, angular: _AngularGalerkin):
        self.layout = angular.layout
        self._dofs = tuple(
            (index, imaginary)
            for index, (_ell, m) in enumerate(self.layout.pairs)
            for imaginary in ((False,) if m == 0 else (False, True))
        )
        key = (self.layout.l_max, self.layout.m_max)
        if key not in self._operator_cache:
            self._operator_cache[key] = self._compile(angular)
        radial_edges, angular_edges = self._operator_cache[key]
        self.radial_edges = tuple(jnp.asarray(value) for value in radial_edges)
        self.angular_edges = tuple(jnp.asarray(value) for value in angular_edges)

    def _pack_numpy(self, coefficients: np.ndarray) -> np.ndarray:
        parts = [
            np.imag(coefficients[..., index]) if imaginary else np.real(coefficients[..., index])
            for index, imaginary in self._dofs
        ]
        return np.stack(parts, axis=-1)

    @staticmethod
    def _edges(matrices: np.ndarray) -> tuple[np.ndarray, ...]:
        nonzero = np.argwhere(np.abs(matrices) > 1.0e-12)
        component, output, input_ = nonzero.T
        values = matrices[component, output, input_]
        return (
            component.astype(np.int32),
            output.astype(np.int32),
            input_.astype(np.int32),
            values,
        )

    def _compile(self, angular: _AngularGalerkin) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        number_dofs = len(self._dofs)
        number_harmonics = self.layout.size
        basis_coefficients = np.zeros((number_dofs, number_harmonics), dtype=np.complex128)
        for dof, (index, imaginary) in enumerate(self._dofs):
            basis_coefficients[dof, index] = 1j if imaginary else 1.0

        reconstruction = np.asarray(angular.reconstruction_basis)
        dtheta_reconstruction = np.asarray(angular.dtheta_reconstruction_basis)
        dphi_reconstruction = np.asarray(angular.dphi_reconstruction_basis)
        projection = np.asarray(angular.projection_basis)
        directions = np.asarray(angular.directions)
        theta_directions = np.asarray(angular.theta_directions)
        phi_directions = np.asarray(angular.phi_directions)
        sin_theta = np.asarray(angular.sin_theta)

        samples = np.real(np.einsum("dh,hq->dq", basis_coefficients, reconstruction))
        dtheta = np.real(np.einsum("dh,hq->dq", basis_coefficients, dtheta_reconstruction))
        dphi = np.real(np.einsum("dh,hq->dq", basis_coefficients, dphi_reconstruction))
        radial_matrices = np.zeros((9, number_dofs, number_dofs))
        angular_matrices = np.zeros_like(radial_matrices)
        for i in range(3):
            for j in range(3):
                component = 3 * i + j
                radial_coefficient = directions[:, i] * directions[:, j]
                theta_coefficient = theta_directions[:, i] * directions[:, j]
                phi_coefficient = phi_directions[:, i] * directions[:, j]
                radial_values = radial_coefficient[None, :] * samples
                angular_values = theta_coefficient[None, :] * dtheta
                angular_values += phi_coefficient[None, :] * dphi / sin_theta[None, :]
                angular_values += ((1.0 if i == j else 0.0) - 3.0 * radial_coefficient)[None, :] * samples
                radial_coefficients = np.einsum("dq,hq->dh", radial_values, projection)
                angular_coefficients = np.einsum("dq,hq->dh", angular_values, projection)
                radial_matrices[component] = self._pack_numpy(radial_coefficients).T
                angular_matrices[component] = self._pack_numpy(angular_coefficients).T
        return self._edges(radial_matrices), self._edges(angular_matrices)

    def _pack(self, coefficients: Array) -> Array:
        parts = [
            jnp.imag(coefficients[..., index, :]) if imaginary else jnp.real(coefficients[..., index, :])
            for index, imaginary in self._dofs
        ]
        return jnp.stack(parts, axis=-2)

    def _unpack(self, coefficients: Array) -> Array:
        output = jnp.zeros(
            (*coefficients.shape[:-2], self.layout.size, coefficients.shape[-1]),
            dtype=jnp.result_type(coefficients.dtype, jnp.complex64),
        )
        for dof, (index, imaginary) in enumerate(self._dofs):
            value = 1j * coefficients[..., dof, :] if imaginary else coefficients[..., dof, :]
            output = output.at[..., index, :].add(value)
        return output

    def _apply(self, coefficients: Array, velocity_gradient: Array, edges: tuple[Array, ...]) -> Array:
        component, output, input_, values = edges
        real_coefficients = self._pack(coefficients)
        gradient_weights = velocity_gradient[..., component // 3, component % 3]
        contributions = gradient_weights[..., None] * real_coefficients[..., input_, :]
        contributions *= values[..., None]
        result = jnp.zeros_like(real_coefficients)
        result = result.at[..., output, :].add(contributions)
        return self._unpack(result)

    def radial(self, coefficients: Array, velocity_gradient: Array) -> Array:
        return self._apply(coefficients, velocity_gradient, self.radial_edges)

    def angular(self, coefficients: Array, velocity_gradient: Array) -> Array:
        return self._apply(coefficients, velocity_gradient, self.angular_edges)

    @property
    def nnz(self) -> int:
        return int(self.radial_edges[3].size + self.angular_edges[3].size)


class IonFrameVlasov:
    """Add prescribed ion-frame kinematics to a ``TzoufrasVlasov`` operator.

    ``velocity_gradient[..., i, j]`` is ``partial_j u_i``.  The supplied
    electric field is in the laboratory frame; ``u_i x B`` is added before the
    peculiar-velocity Lorentz push.  ``material_acceleration`` must use the
    same normalized acceleration units as the electric-force coefficient.
    """

    def __init__(self, vlasov: TzoufrasVlasov):
        self.vlasov = vlasov
        self.layout = vlasov.layout
        self.angular = _AngularGalerkin(self.layout)
        self.sparse_angular = _SparseAngularCoupling(self.angular)

    def velocity_gradient(self, ion_velocity: Array) -> Array:
        """Return ``partial_j u_i`` with unresolved z derivatives set to zero."""

        if ion_velocity.shape[-1] != 3:
            raise ValueError("ion_velocity must have a final component axis of length 3")
        derivative_x = jnp.real(self.vlasov.spatial_derivative(ion_velocity, axis=0))
        derivative_y = jnp.real(self.vlasov.spatial_derivative(ion_velocity, axis=1))
        return jnp.stack((derivative_x, derivative_y, jnp.zeros_like(derivative_x)), axis=-1)

    def bulk_advection(self, f: Array, ion_velocity: Array, dfdz: Array | None = None) -> Array:
        """Conservative periodic transport ``-div_x(u_i f)``."""

        if ion_velocity.shape != (*f.shape[:-2], 3):
            raise ValueError("ion_velocity shape must match the spatial VFP grid and have three components")
        flux_x = ion_velocity[..., 0, None, None] * f
        flux_y = ion_velocity[..., 1, None, None] * f
        result = -self.vlasov.spatial_derivative(flux_x, axis=0)
        result -= self.vlasov.spatial_derivative(flux_y, axis=1)
        if dfdz is not None:
            result -= ion_velocity[..., 2, None, None] * dfdz
        return result

    def deformation_reference(self, f: Array, velocity_gradient: Array) -> Array:
        """Dense Galerkin oracle for ``div_c((grad u_i)c f)``."""

        expected_shape = (*f.shape[:-2], 3, 3)
        if velocity_gradient.shape != expected_shape:
            raise ValueError(f"velocity_gradient must have shape {expected_shape}")

        samples = self.angular.reconstruct(f)
        dtheta, dphi = self.angular.angular_derivatives(f)
        gradient_times_direction = jnp.einsum("...ij,qj->...qi", velocity_gradient, self.angular.directions)
        radial_coefficient = jnp.einsum("qi,...qi->...q", self.angular.directions, gradient_times_direction)
        theta_coefficient = jnp.einsum("qi,...qi->...q", self.angular.theta_directions, gradient_times_direction)
        phi_coefficient = jnp.einsum("qi,...qi->...q", self.angular.phi_directions, gradient_times_direction)
        trace = jnp.trace(velocity_gradient, axis1=-2, axis2=-1)

        face_speed = jnp.arange(self.vlasov.v.size + 1, dtype=self.vlasov.v.dtype) * self.vlasov.dv
        interior_flux = (
            radial_coefficient[..., None] * face_speed[1:-1] ** 3 * 0.5 * (samples[..., 1:] + samples[..., :-1])
        )
        zero_flux = jnp.zeros_like(samples[..., :1])
        radial_flux = jnp.concatenate((zero_flux, interior_flux, zero_flux), axis=-1)
        radial_divergence = (radial_flux[..., 1:] - radial_flux[..., :-1]) / (self.vlasov.v**2 * self.vlasov.dv)

        angular_divergence = theta_coefficient[..., None] * dtheta
        angular_divergence += phi_coefficient[..., None] * dphi / self.angular.sin_theta[..., None]
        angular_divergence += (trace[..., None] - 3.0 * radial_coefficient)[..., None] * samples
        return self.angular.project(radial_divergence + angular_divergence)

    def deformation(self, f: Array, velocity_gradient: Array) -> Array:
        """Return ``div_c((grad u_i)c f)`` using sparse harmonic couplings."""

        expected_shape = (*f.shape[:-2], 3, 3)
        if velocity_gradient.shape != expected_shape:
            raise ValueError(f"velocity_gradient must have shape {expected_shape}")

        face_speed = jnp.arange(self.vlasov.v.size + 1, dtype=self.vlasov.v.dtype) * self.vlasov.dv
        face_average = 0.5 * (f[..., 1:] + f[..., :-1])
        interior_flux = face_speed[1:-1] ** 3 * self.sparse_angular.radial(face_average, velocity_gradient)
        zero_flux = jnp.zeros_like(f[..., :1])
        radial_flux = jnp.concatenate((zero_flux, interior_flux, zero_flux), axis=-1)
        radial_divergence = (radial_flux[..., 1:] - radial_flux[..., :-1]) / (self.vlasov.v**2 * self.vlasov.dv)
        return radial_divergence + self.sparse_angular.angular(f, velocity_gradient)

    def frame_acceleration(self, f: Array, material_acceleration: Array) -> Array:
        """Return ``D_t u_i . grad_c(f)`` using the existing force operator."""

        return self.vlasov.electric(f, material_acceleration)

    def __call__(
        self,
        f: Array,
        electric_field: Array,
        magnetic_field: Array,
        ion_velocity: Array,
        *,
        velocity_gradient: Array | None = None,
        material_acceleration: Array | None = None,
        dfdz: Array | None = None,
    ) -> Array:
        if velocity_gradient is None:
            velocity_gradient = self.velocity_gradient(ion_velocity)
        if material_acceleration is None:
            material_acceleration = jnp.zeros_like(ion_velocity)
        rest_frame_electric = electric_field + jnp.cross(ion_velocity, magnetic_field)
        result = self.vlasov(f, rest_frame_electric, magnetic_field, dfdz=dfdz)
        result += self.bulk_advection(f, ion_velocity, dfdz=dfdz)
        result += self.deformation(f, velocity_gradient)
        result += self.frame_acceleration(f, material_acceleration)
        for index, (_ell, m) in enumerate(self.layout.pairs):
            if m == 0:
                result = result.at[..., index, :].set(jnp.real(result[..., index, :]))
        return result
