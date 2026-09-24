"""Two-dimensional spherical-harmonic Vlasov--Fokker--Planck solver."""

from importlib import import_module

_LAZY_ATTRIBUTES = {
    "AnisotropicCollisions": ("adept.vfp2d.collisions", "AnisotropicCollisions"),
    "BaseVFP2D": ("adept.vfp2d.base", "BaseVFP2D"),
    "CollisionStep": ("adept.vfp2d.collisions", "CollisionStep"),
    "CoupledIonKineticStep": ("adept.vfp2d.coupling", "CoupledIonKineticStep"),
    "ElectronIonExchange": ("adept.vfp2d.exchange", "ElectronIonExchange"),
    "ElectronPressureCoupling": ("adept.vfp2d.pressure", "ElectronPressureCoupling"),
    "Grid": ("adept.vfp2d.grid", "Grid"),
    "HarmonicLayout": ("adept.vfp2d.harmonics", "HarmonicLayout"),
    "HouLiFilter2D": ("adept.vfp2d.harmonics", "HouLiFilter2D"),
    "IonEuler2D": ("adept.vfp2d.hydro", "IonEuler2D"),
    "IonFrameVlasov": ("adept.vfp2d.moving_frame", "IonFrameVlasov"),
    "IonMagneticCoupling": ("adept.vfp2d.magnetic", "IonMagneticCoupling"),
    "KineticOhm2D": ("adept.vfp2d.ohm", "KineticOhm2D"),
    "KineticOhmStep": ("adept.vfp2d.vector_field", "KineticOhmStep"),
    "Maxwell2D": ("adept.vfp2d.vector_field", "Maxwell2D"),
    "OSHUNImplicitStep": ("adept.vfp2d.vector_field", "OSHUNImplicitStep"),
    "SpectralPoisson2D": ("adept.vfp2d.vector_field", "SpectralPoisson2D"),
    "SplitStepVFP2D": ("adept.vfp2d.vector_field", "SplitStepVFP2D"),
    "TzoufrasVlasov": ("adept.vfp2d.harmonics", "TzoufrasVlasov"),
    "VFP2DBuilder": ("adept.vfp2d.builder", "VFP2DBuilder"),
    "VelocityFrameRemap": ("adept.vfp2d.exchange", "VelocityFrameRemap"),
    "VlasovMaxwell": ("adept.vfp2d.vector_field", "VlasovMaxwell"),
    "cartesian_l2": ("adept.vfp2d.harmonics", "cartesian_l2"),
    "complex_to_real": ("adept.vfp2d.harmonics", "complex_to_real"),
    "conservative_f00_positivity": ("adept.vfp2d.harmonics", "conservative_f00_positivity"),
    "conserved_to_primitive": ("adept.vfp2d.hydro", "conserved_to_primitive"),
    "coupled_invariants": ("adept.vfp2d.coupling", "coupled_invariants"),
    "current": ("adept.vfp2d.harmonics", "current"),
    "density": ("adept.vfp2d.harmonics", "density"),
    "electron_kinetic_energy_density": ("adept.vfp2d.exchange", "electron_kinetic_energy_density"),
    "electron_momentum_density": ("adept.vfp2d.exchange", "electron_momentum_density"),
    "electron_pressure_tensor": ("adept.vfp2d.pressure", "electron_pressure_tensor"),
    "euler_flux": ("adept.vfp2d.hydro", "euler_flux"),
    "hllc_flux": ("adept.vfp2d.hydro", "hllc_flux"),
    "nernst_velocity": ("adept.vfp2d.harmonics", "nernst_velocity"),
    "primitive_to_conserved": ("adept.vfp2d.hydro", "primitive_to_conserved"),
    "project_current_moment": ("adept.vfp2d.ohm", "project_current_moment"),
    "real_to_complex": ("adept.vfp2d.harmonics", "real_to_complex"),
    "scalar_velocity_moment": ("adept.vfp2d.harmonics", "scalar_velocity_moment"),
    "tensor_velocity_moment": ("adept.vfp2d.harmonics", "tensor_velocity_moment"),
    "vector_velocity_moment": ("adept.vfp2d.harmonics", "vector_velocity_moment"),
}

__all__ = list(_LAZY_ATTRIBUTES)


def __getattr__(name):
    if name not in _LAZY_ATTRIBUTES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _LAZY_ATTRIBUTES[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
