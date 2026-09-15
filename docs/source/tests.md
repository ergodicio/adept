# Tests

## Running the tests

The test dependencies come with the `dev` extra, so no separate install is needed:

```bash
uv run pytest
```

Long-running parameter sweeps are marked `slow` and excluded by default (see `addopts` in
`pyproject.toml`). Run them explicitly with:

```bash
uv run pytest -m slow
```

Some suites include relatively expensive simulations. To run one suite, or to see example usage of a
particular solver, point pytest at a directory or use `-k`:

```bash
uv run pytest tests/test_vlasov1d
uv run pytest -k landau_damping
```

The tests double as the most reliable examples in the repo — each one builds a config and drives a
solver through the `ergoExo` lifecycle.

## Test Suite

Tests are organized one directory per solver.

| Directory | Covers |
|---|---|
| `tests/test_base` | `ergoExo` lifecycle, MLflow logging, shared functions, Chang-Cooper differencing |
| `tests/test_vlasov1d` | Landau damping, absorbing wave boundaries, EM dispersion, multispecies, Fokker-Planck conservation, Boltzmann electrons, config validation and regression |
| `tests/test_vlasov1d2v` | Equivalence with the 1D solver in the separable limit, conservation, and the cylindrical Landau operator |
| `tests/test_vlasov2d` | Landau damping, EM dispersion, gyrorotation, distributed initialization |
| `tests/test_vfp1d` | Fokker-Planck models and relaxation, heating, Epperlein-Haines transport coefficients, spherical geometry |
| `tests/test_lpse2d` | EPW frequency, TPD threshold, speckle, dealiasing, pretrained-driver loading |
| `tests/test_spectrax1d` | Landau damping, Maxwell solver, shift and Lorentz operators |
| `tests/test_hermite_legendre_1d` | Conservation, streaming, Landau damping, linear advection, structured split/IMEX integrators |
| `tests/test_hermite_poisson_1d` | Collisions, E-field coupling, drivers, integrators, filtering, linear response |
| `tests/test_pic1d` | Bohm-Gross dispersion, Landau damping, two-stream instability |
| `tests/test_tf1d` | Bohm-Gross and kinetic resonance (forward and backward pass), Landau damping, agreement with a Vlasov run |

## Continuous integration

`.github/workflows/cpu-tests.yaml` runs the suites on every pull request. Solver suites are gated by
a `dorny/paths-filter` step so that touching one solver runs only its tests, while touching shared
code (`adept/_base_.py`, `adept/utils.py`, `pyproject.toml`, …) runs everything. A push to `main` or
a manual dispatch always runs the full set.

```{note}
The `test_vlasov1d2v`, `test_vlasov2d`, and `test_tf1d` suites do not currently have a CI job and so
are only run locally.
```
