# Running a simulation

Simulations are defined by YAML config files.
To run a single simulation,
```
uv run run.py --cfg path_to_my_config
```

## Module-Specific Configuration Reference

- [Vlasov-1D](source/solvers/vlasov1d/config.md) - 1D Vlasov-Poisson/Maxwell solver with Fokker-Planck collisions
- [Vlasov-1D2V](source/solvers/vlasov1d2v/config.md) - 1D2V Vlasov-Poisson-Fokker-Planck in cylindrical velocity space, with a full-geometry Coulomb collision operator
- [Vlasov-2D](source/solvers/vlasov2d/config.md) - 2D2V Vlasov-Maxwell solver
- [VFP-1D](source/solvers/vfp1d/config.md) - Vlasov-Fokker-Planck electron transport solver
- [VFP-2D](source/solvers/vfp2d/config.md) - 2D3P arbitrary-spherical-harmonic Vlasov-Maxwell-Fokker-Planck solver
- [LPSE-2D (Envelope-2D)](source/solvers/lpse2d/config.md) - 2D laser-plasma envelope solver for TPD/SRS
- [Spectrax-1D](source/solvers/spectrax1d/config.md) - 1D Hermite-Fourier Vlasov-Maxwell solver
- [Hermite-Legendre-1D](source/solvers/hermite_legendre_1d/config.md) - 1D-1V mixed Hermite-Legendre electrostatic Vlasov-Poisson solver
- [PIC-1D](source/solvers/pic1d/config.md) - 1D1V electrostatic particle-in-cell solver
- [Two-Fluid-1D](source/solvers/tf1d/config.md) - 1D electrostatic two-fluid solver with kinetic closure and particle trapping
- [OSIRIS wrapper](source/solvers/osiris/config.md) - runs the external OSIRIS PIC code from a native input deck

See the [full documentation](https://ergodicio.github.io/adept/) for detailed guides and API reference.
