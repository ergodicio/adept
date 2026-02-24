# Running a simulation

Simulations are defined by YAML config files.
To run a single simulation,
```
uv run run.py --cfg path_to_my_config
```

## Module-Specific Configuration Reference

- [Vlasov-1D](source/solvers/vlasov1d/config.md) - 1D Vlasov-Poisson/Maxwell solver with Fokker-Planck collisions
- [Vlasov-2D](source/solvers/vlasov2d/config.md) - 2D2V Vlasov-Maxwell solver
- [LPSE-2D (Envelope-2D)](source/solvers/lpse2d/config.md) - 2D laser-plasma envelope solver for TPD/SRS
- [Spectrax-1D](source/solvers/spectrax1d/config.md) - 1D Hermite-Fourier Vlasov-Maxwell solver

See the [full documentation](https://ergodicio.github.io/adept/) for detailed guides and API reference.
