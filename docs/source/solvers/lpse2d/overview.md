# LPSE 2D (Envelope-2D) Solver

Example decks live in `configs/envelope-2d/`. To run one:

```bash
uv run run.py --cfg configs/envelope-2d/tpd
```

## Equations and Quantities

These equations model the evolution and interaction of the complex envelopes of light waves and plasma waves. This is faster than modeling the plasma waves using a fluid or kinetic solver along with modeling the light waves.

### Note on Pump Depletion

One can solve these equations with or without "pump depletion". "Pump depletion" is the effect of the plasma waves on the light waves. By default the pump is prescribed analytically (an external driver for the plasma waves), which is adequate below the absolute instability threshold; above it the plasma-wave and Raman fields grow without bound because nothing depletes the pump.

For SRS, setting `terms.light.pump_depletion: true` evolves the pump `E0` with the same finite-difference envelope solver as the Raman light, sourced by a two-point boundary injector at the low-density side and coupled to the EPW through $-i e/(4 \omega_1 m_e) (\nabla^2 \phi) \mathbf{E}_1$ (the conjugate partner of the Raman coupling; the pair conserves $\int |E_0|^2 + |E_1|^2 + |\nabla\phi|^2$ together with the EPW source). This enables true transmission/absorption diagnostics and saturation of above-threshold runs.

### SRS diagnostics

With `terms.epw.source.srs` on, the default time series includes OSIRIS-comparable channels (fields converted to $m_e c \omega_0/e$, lengths to $c/\omega_0$, fluxes normalized to the nominal incident flux):

| series | meaning | OSIRIS scan2 counterpart |
|---|---|---|
| `epw_energy` | $\frac{1}{4}\,dx \sum_x \langle\|E_{epw}\|^2\rangle_y$ (cycle-averaged field energy) | `W(t) = 1/2 dx sum e1^2` |
| `epw_dissipation` | total EPW energy handed to electrons per ps (Landau + collisional, the solver's own rates; includes the kinetic half) | absorbed fraction / hot-electron source |
| `epw_boundary_loss` | total EPW energy removed by the absorbing boundaries per ps | — |
| `incident_flux`, `transmitted_flux` | pump flux at probes near each edge | `incident_t`, `T_t` |
| `reflected_flux`, `backrefl_flux` | Raman flux leaving left / right | `R_t` |
| `reflectivity`, `e1_sq` | legacy probes (unchanged) | — |

`post_process` logs scalar metrics with the same names and definitions as `osiris_lpi` (`laser_reflectivity`, `laser_transmissivity`, `laser_absorbed_frac`, per-quarter `_seg{i}of4` variants, `epw_growth_rate` per $\omega_0$ with the identical automated fit window, `electron_energy_frac_final`, and the `laser_incident_flux_ratio` health check).

### Electron Plasma Waves

$$
\nabla \cdot \left[ i \left(\frac{\partial}{\partial t} + \nu_e^{\circ} \right) + \frac{3 v_{te}^2}{2 \omega_{p0}} \nabla^2 + \frac{\omega_{p0}}{2}\left(1-\frac{n_b(x)}{n_0}\right) \right] \textbf{E}_h = S_{TPD} + S_h
$$

### Two Plasmon Decay

$$
S_{\text{TPD}} \equiv \frac{e}{8 \omega_{p0} m_e} \frac{n_b(x)}{n_0} \nabla \cdot [\nabla (\textbf{E}_0 \cdot \textbf{E}_h^*) - \textbf{E}_0 \nabla\cdot \textbf{E}_h^*] e^{-i (\omega_0 - 2 \omega_{p0})t}
$$

### Laser Driver

We only have a plane wave implementation for now:

$$
E_0(t, x, y) = \sum_j^{N_c} A_j ~ \exp(-i k_0 x - i \omega_0 \Delta\omega_j ~ t + \phi_j)
$$

## Boundary Conditions

Set per axis under `terms.epw.boundary`:

| Value | Behaviour |
|---|---|
| `absorbing` | A damping layer of width `grid.boundary_width` is applied at both ends of that axis. The mask is $\exp(-\alpha \, dt \, (1 - \text{env}_x \text{env}_y))$ with $\alpha =$ `grid.boundary_abs_coeff` and a tanh ramp over `boundary_width / 5`, so the envelope is attenuated smoothly rather than clipped. |
| anything else | **Periodic** — the underlying representation is spectral, so this is the default behaviour when no absorbing layer is requested. |

`x` and `y` are configured independently, which is the usual arrangement for TPD: absorbing along the
density gradient, periodic transverse to it. The $k=0$ mode is masked out of the spectral operators.

## Forcing and Drivers

Because pump depletion is not implemented, **the laser is pure forcing** — it drives the plasma waves
and is never itself depleted. That is the central approximation of this solver and the reason it is
only valid below the absolute instability threshold.

| Term | Role |
|---|---|
| $E_0$ (laser driver) | A sum of $N_c$ plane-wave components, each with amplitude $A_j$, frequency offset $\Delta\omega_j$, and phase $\phi_j$. Bandwidth models (SSD, CPP speckle) are built by populating these components — see the example decks `tpd-static-cpp-speckle.yaml` and `tpd-dynamic-ssd-speckle.yaml`. |
| $S_\text{TPD}$ | The two-plasmon-decay source coupling $E_0$ to the plasma-wave envelope. |
| $S_h$ | A noise/seed source for the plasma waves. |
| $\nu_e^\circ$ | Landau damping of the plasma-wave envelope. |

## What Gets Saved

**`binary/`**:

| File | Contents |
|---|---|
| `fields.xr` | The real-space envelope fields over time |
| `k-fields.xr` | The same in wavenumber space — this is where TPD growth is measured |
| `series.xr` | Scalar time series |

**`plots/`**, per field `k`:

| File | Contents |
|---|---|
| `<k>_x.png`, `<k>_x_r.png` | Spatial profiles, magnitude and real part |
| `<k>.png`, `log-<k>.png`, `real-<k>.png` | Slices, linear / log / real part |
| `spacetime-<k>.png`, `spacetime-log-<k>.png`, `spacetime-real-<k>.png` | Space-time plots |

The log-scaled space-time plots are the ones to read for instability growth rates.

## Configuration Reference

See the [Configuration Reference](config.md) for complete YAML schema documentation.
