# ADEPT Architecture

- ADEPT solvers are packaged into `ADEPTModule`s and run via the `ergoExo`
  Code pointer: `adept/_base_.py`
- The `ergoExo` manages creation of an MLflow "run" and calls lifecycle methods on the `ADEPTModule` to perform logging of configuration, parameters, and run artifacts.
  Code pointer: `adept/_base_.py: ergoExo#_setup_()`
- The different `ADEPTModule`s are defined in subdirectories of `adept`, for example `_vlasov1d`, `_lpse2d`, etc.
- `ADEPTModule`s wrap a `diffrax` differential equation solver. The RHS of the ODE (often a discretized PDE) is typically found in a file named `vector_field.py`, in the class `VectorField`. For example, the Vlasov1D RHS is defined in `adept/_vlasov1d/solvers/vector_field.py`.

## Module Documentation

See the [full documentation](https://ergodicio.github.io/adept/) for detailed solver guides.

Quick links to configuration references:
- [Vlasov-1D Config](source/solvers/vlasov1d/config.md)
- [VFP-1D Config](source/solvers/vfp1d/config.md)
- [Vlasov-2D Config](source/solvers/vlasov2d/config.md)
- [LPSE-2D Config](source/solvers/lpse2d/config.md)
- [Spectrax-1D Config](source/solvers/spectrax1d/config.md)
