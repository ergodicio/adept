"""Expose Vlasov1D numerical operators as explicit JAX PyTrees.

The legacy constructors still perform host-side setup. Unflattening restores
only their numerical state, without repeating normalization, quadrature, or
configuration parsing. Arrays, species data, drivers, and nested operators are
dynamic children; algorithm choices and sharding meshes are static metadata.
"""

import jax.tree_util as jtu

from adept._vlasov1d.solvers.pushers import field, fokker_planck, vlasov
from adept._vlasov1d.solvers.vector_field import (
    LeapfrogIntegrator,
    SixthOrderHamIntegrator,
    StrangIntegrator,
    TimeIntegrator,
    VlasovMaxwell,
    VlasovPoissonFokkerPlanck,
)


def _register(cls, data_fields):
    def flatten(operator):
        # Optional operators, such as the Hou-Li filter, need not be constructed
        # when disabled. Their presence is part of the static tree structure.
        names = tuple(name for name in data_fields if hasattr(operator, name))
        children = tuple((jtu.GetAttrKey(name), getattr(operator, name)) for name in names)
        metadata = tuple(sorted((name, value) for name, value in vars(operator).items() if name not in names))
        return children, (names, metadata)

    def unflatten(aux, children):
        names, metadata = aux
        operator = object.__new__(cls)
        for name, value in (*metadata, *zip(names, children, strict=True)):
            setattr(operator, name, value)
        return operator

    jtu.register_pytree_with_keys(cls, flatten, unflatten)


_register(field.LongitudinalElectricFieldDriver, ("xax", "drivers"))
_register(
    field.TransverseCurrentSourceDriver,
    ("xax", "drivers", "point_source_masks", "point_source_scales", "dx", "c"),
)
_register(field.WaveSolver, ("dx", "c", "c_sq", "dt", "const", "one_over_const"))
_register(field.SpectralPoissonSolver, ("one_over_kx", "species_grids", "species_params", "static_charge_density"))
_register(field.BoltzmannPoissonSolver, ("kx", "species_grids", "species_params", "Te", "lambda_De"))
_register(field.AmpereSolver, ("species_grids", "species_params"))
_register(field.HampereSolver, ("kx", "one_over_ikx", "species_grids", "species_params", "vx", "dv", "charge"))
_register(field.ElectricFieldSolver, ("es_field_solver", "dx"))

for pusher in (
    vlasov.VelocityExponential,
    vlasov.VelocityCubicSpline,
    vlasov.VelocityLagrange7,
    vlasov.VelocityPFC3,
    vlasov.VelocitySLWENO5,
):
    _register(pusher, ("species_grids", "species_params"))
_register(vlasov.SpaceExponential, ("kx_real", "species_grids"))
_register(vlasov.SpacePFC3, ("dx", "species_grids"))
_register(vlasov.SpaceSLWENO5, ("dx", "species_grids"))
_register(vlasov.HouLiFilter, ("filter_x",))

_register(fokker_planck.Krook, ("f_mx", "dv"))
_register(fokker_planck.Collisions, ("fp_model", "fp_scheme", "krook", "v", "dv", "v_edge"))

_integrator_fields = ("field_solve", "species_grids", "species_params", "edfdv", "vdfdx", "dt", "dt_array")
_register(TimeIntegrator, _integrator_fields)
_register(LeapfrogIntegrator, _integrator_fields)
_register(StrangIntegrator, _integrator_fields)
_register(SixthOrderHamIntegrator, (*_integrator_fields, "a1", "a2", "a3", "D1", "D2", "D3"))
_register(VlasovPoissonFokkerPlanck, ("dt", "vlasov_poisson", "fp", "hou_li_filter"))
_register(
    VlasovMaxwell,
    (
        "x",
        "electron_dv",
        "electron_charge",
        "nu_fp_prof",
        "nu_K_prof",
        "vpfp",
        "wave_solver",
        "dt",
        "ey_driver",
        "ex_driver",
        "ex_stochastic",
    ),
)
