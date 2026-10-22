"""Expose the existing numerical operators as explicit JAX PyTrees.

The legacy operators remain mutable Python objects for compatibility. Registration
reconstructs them without running host-side constructors (normalization, quadrature,
and validation). Every array and nested numerical operator is a dynamic child;
scalar settings, harmonic topology, and mesh descriptions are static metadata.
"""

import jax.tree_util as jtu

from adept.vfp1d.fokker_planck import FLMCollisions
from adept.vfp2d.collisions import AnisotropicCollisions, CollisionStep
from adept.vfp2d.coupling import CoupledIonKineticStep
from adept.vfp2d.exchange import ElectronIonExchange, VelocityFrameRemap
from adept.vfp2d.harmonics import HouLiFilter2D, TzoufrasVlasov
from adept.vfp2d.magnetic import IonMagneticCoupling
from adept.vfp2d.moving_frame import IonFrameVlasov, _AngularGalerkin, _SparseAngularCoupling
from adept.vfp2d.ohm import KineticOhm2D
from adept.vfp2d.pressure import ElectronPressureCoupling
from adept.vfp2d.reservoir import DrivenReservoirStep
from adept.vfp2d.vector_field import KineticOhmStep, Maxwell2D, OSHUNImplicitStep, SplitStepVFP2D, VlasovMaxwell


def _register(cls, data_fields):
    def flatten(operator):
        children = tuple((jtu.GetAttrKey(name), getattr(operator, name)) for name in data_fields)
        metadata = tuple(sorted((name, value) for name, value in vars(operator).items() if name not in data_fields))
        return children, metadata

    def unflatten(metadata, children):
        operator = object.__new__(cls)
        for name, value in (*metadata, *zip(data_fields, children, strict=True)):
            setattr(operator, name, value)
        return operator

    jtu.register_pytree_with_keys(cls, flatten, unflatten)


_register(FLMCollisions, ("grid", "a1", "a2", "b1", "b2", "b3", "b4"))
_register(AnisotropicCollisions, ("operator",))
_register(CollisionStep, ("isotropic", "anisotropic"))
_register(TzoufrasVlasov, ("v", "kx", "ky", "streaming_speed"))
_register(HouLiFilter2D, ("filter_x", "filter_y"))
_register(Maxwell2D, ("kx", "ky"))
_register(VlasovMaxwell, ("vlasov", "maxwell", "v", "streaming_speed"))
_register(SplitStepVFP2D, ("rhs", "collisions"))
_register(
    OSHUNImplicitStep,
    ("vlasov", "maxwell", "v", "collisions", "streaming_speed", "spatial_filter"),
)
_register(KineticOhmStep, ("vlasov", "maxwell", "ohm", "v", "collisions", "spatial_filter", "ion_frame"))
_register(KineticOhm2D, ("v", "kx", "ky"))
_register(
    _AngularGalerkin,
    (
        "basis",
        "reconstruction_basis",
        "dtheta_reconstruction_basis",
        "dphi_reconstruction_basis",
        "projection_basis",
        "coefficient_scale",
        "directions",
        "theta_directions",
        "phi_directions",
        "sin_theta",
    ),
)
_register(_SparseAngularCoupling, ("radial_edges", "angular_edges"))
_register(IonFrameVlasov, ("vlasov", "angular", "sparse_angular"))
_register(ElectronIonExchange, ("v",))
_register(VelocityFrameRemap, ("ion_frame", "v"))
_register(ElectronPressureCoupling, ("ion_frame", "v"))
_register(IonMagneticCoupling, ("maxwell",))
_register(CoupledIonKineticStep, ("electron_step", "exchange", "pressure", "magnetic", "frame_remap"))
_register(DrivenReservoirStep, ("step", "electrons", "target", "rate"))
