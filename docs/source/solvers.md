# Available Solvers

ADEPT provides several solver modules for different plasma physics applications. The `solver:` key
at the top of a configuration file selects which one runs.

| `solver:` key | Module | Description |
| --- | --- | --- |
| `vlasov-1d` | [Vlasov-1D](solvers/vlasov1d/overview.md) | 1D1V Vlasov-Poisson/Maxwell with Fokker-Planck collisions |
| `vlasov-1d-iaw` | [Vlasov-1D](solvers/vlasov1d/overview.md) | Vlasov-1D with kinetic ions, Boltzmann electrons, and IAW turbulence diagnostics |
| `vlasov-1d2v` | [Vlasov-1D2V](solvers/vlasov1d2v/overview.md) | 1D2V in cylindrical velocity space with a full-geometry Coulomb collision operator |
| `vlasov-2d` | [Vlasov-2D](solvers/vlasov2d/overview.md) | 2D2V Vlasov-Maxwell |
| `vfp-1d` | [VFP-1D](solvers/vfp1d/overview.md) | Vlasov-Fokker-Planck electron transport |
| `envelope-2d` | [LPSE-2D](solvers/lpse2d/overview.md) | 2D laser-plasma envelope solver |
| `spectrax-1d` | [Spectrax-1D](solvers/spectrax1d/overview.md) | Hermite-Fourier Vlasov-Maxwell |
| `hermite-epw-1d` | [Spectrax-1D](solvers/spectrax1d/overview.md) | Spectrax-1D with electron plasma wave diagnostics |
| `hermite-maxwell-1d` | [Spectrax-1D](solvers/spectrax1d/overview.md) | Spectrax-1D with EM dispersion/absorption diagnostics |
| `hermite-legendre-1d` | [Hermite-Legendre-1D](solvers/hermite_legendre_1d/overview.md) | 1D1V mixed Hermite-Legendre electrostatic Vlasov-Poisson |
| `pic-1d` | [PIC-1D](solvers/pic1d/overview.md) | 1D1V electrostatic particle-in-cell |
| `tf-1d` | [Two-Fluid 1D](solvers/tf1d/overview.md) | 1D warm two-fluid Poisson with kinetic closures |

## Kinetic Solvers

### [Vlasov-1D](solvers/vlasov1d/overview.md)

1D1V Vlasov-Poisson/Maxwell solver with Fokker-Planck collisions. Ideal for studying electron plasma waves, Landau damping, and wave-particle interactions.

- [Overview & Equations](solvers/vlasov1d/overview.md)
- [Configuration Reference](solvers/vlasov1d/config.md)

Two additional `solver:` keys select problem-specific variants of this module:
`vlasov-1d-iaw` adds kinetic ions with a linearized Boltzmann electron closure and ion-acoustic
turbulence diagnostics (see
[boltzmann_electrons](solvers/vlasov1d/config.md#boltzmann_electrons-kinetic-ions-with-adiabatic-electrons)).

### [Vlasov-1D2V](solvers/vlasov1d2v/overview.md)

1D2V solver in cylindrical velocity space $(v_\parallel, v_\perp)$. The field solve reuses the
Vlasov-1D machinery by feeding it velocity marginals; the point of the extra dimension is a
full-geometry linearized Coulomb operator with separable speed and pitch-angle channels, which a 1V
velocity space cannot represent.

- [Overview & Equations](solvers/vlasov1d2v/overview.md)
- [Configuration Reference](solvers/vlasov1d2v/config.md)

### [Vlasov-2D](solvers/vlasov2d/overview.md)

2D2V Vlasov-Maxwell solver for electromagnetic simulations in two spatial dimensions.

- [Overview & Equations](solvers/vlasov2d/overview.md)
- [Configuration Reference](solvers/vlasov2d/config.md)

### [VFP-1D](solvers/vfp1d/overview.md)

Vlasov-Fokker-Planck solver for electron transport over collisional time-scales. Uses a spherical harmonic expansion with staggered spatial grid and full FLM collision operator.

- [Overview & Equations](solvers/vfp1d/overview.md)
- [Configuration Reference](solvers/vfp1d/config.md)

## Spectral Solvers

These represent velocity space with a spectral basis rather than a grid, so the state is a set of
mode coefficients instead of a sampled distribution function.

### [Spectrax-1D](solvers/spectrax1d/overview.md)

Hermite-Fourier spectral Vlasov-Maxwell solver. Hermite in velocity, Fourier in space, with an
exponential (Lawson-RK4) integrator that removes the linear stiffness. Three `solver:` keys share
this module: `spectrax-1d`, `hermite-epw-1d`, and `hermite-maxwell-1d`.

- [Overview & Equations](solvers/spectrax1d/overview.md)
- [Configuration Reference](solvers/spectrax1d/config.md)

### [Hermite-Legendre-1D](solvers/hermite_legendre_1d/overview.md)

1D1V electrostatic Vlasov-Poisson using two velocity bases at once — Hermite for the near-Maxwellian
bulk, Legendre on a bounded window for strongly non-Maxwellian features such as beams and plateaus.

- [Overview & Equations](solvers/hermite_legendre_1d/overview.md)
- [Configuration Reference](solvers/hermite_legendre_1d/config.md)

## Particle Solvers

### [PIC-1D](solvers/pic1d/overview.md)

1D1V electrostatic particle-in-cell solver. Built as a particle-based twin of Vlasov-1D — it accepts
the same units, density, driver, and save blocks, so the same deck can be run both ways and
compared.

- [Overview & Equations](solvers/pic1d/overview.md)
- [Configuration Reference](solvers/pic1d/config.md)

## Envelope Solvers

### [LPSE-2D (Envelope-2D)](solvers/lpse2d/overview.md)

2D laser-plasma envelope solver for modeling laser-plasma instabilities including Two-Plasmon Decay (TPD) and Stimulated Raman Scattering (SRS).

- [Overview & Equations](solvers/lpse2d/overview.md)
- [Configuration Reference](solvers/lpse2d/config.md)

## Fluid Solvers

### [Two-Fluid 1D](solvers/tf1d/overview.md)

1D warm two-fluid Poisson solver with kinetic closures — tabulated kinetic dispersion, Landau
damping, and a particle-trapping model. Much cheaper than the kinetic solvers, and the module used
for the machine-learned-closure work.

- [Overview & Equations](solvers/tf1d/overview.md)
- [Configuration Reference](solvers/tf1d/config.md)
