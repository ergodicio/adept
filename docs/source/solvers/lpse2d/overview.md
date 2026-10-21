# LPSE 2D (Envelope-2D) Solver

Example decks live in `configs/envelope-2d/`. To run one:

```bash
uv run run.py --cfg configs/envelope-2d/tpd
```

## Equations and Quantities

These equations model the evolution and interaction of the complex envelopes of light waves and plasma waves. This is faster than modeling the plasma waves using a fluid or kinetic solver along with modeling the light waves.

### Note on Pump Depletion

One can solve these equations with or without "pump depletion". "Pump depletion" is the effect of the plasma waves on the light waves. By default the pump is prescribed analytically (an external driver for the plasma waves), which is adequate below the absolute instability threshold; above it the plasma-wave and Raman fields grow without bound because nothing depletes the pump.

Setting `terms.light.pump_depletion: true` evolves the pump `E0` with the finite-difference envelope solver and a two-point boundary injector at the low-density side. Every active instability contributes its reciprocal pump term:

- SRS contributes $-i e/(4 \omega_1 m_e) (\nabla^2 \phi) \mathbf{E}_1$.
- TPD contributes $i e/(4 \omega_0 m_e) e^{i(\omega_0-2\omega_{p0})t} E_{h,y}\nabla\!\cdot\!\mathbf{E}_h$ to the y-polarized pump. This is the discrete reciprocal of the existing TPD source and conserves $\int[|E_0|^2+(\omega_{p0}/\omega_0)|\mathbf{E}_h|^2]$ for the coupling terms alone.

Both terms are included when TPD and SRS are enabled together. Dynamic pump evolution enables true transmission/absorption diagnostics and provides the physical saturation channel needed above threshold.

### SRS diagnostics

With `terms.epw.source.srs` on, the default time series includes OSIRIS-comparable channels (fields converted to $m_e c \omega_0/e$, lengths to $c/\omega_0$, fluxes normalized to the nominal incident flux):

| series | meaning | OSIRIS scan2 counterpart |
|---|---|---|
| `epw_energy` | $\frac{1}{4}\,dx \sum_x \langle\|E_{epw}\|^2\rangle_y$ (cycle-averaged field energy) | `W(t) = 1/2 dx sum e1^2` |
| `epw_dissipation` | total EPW energy handed to electrons per ps (Landau + collisional, the solver's own rates -- with `terms.epw.solver: combined` the Landau rate plus the combined field's Raman absorption `nu_R (n/n_env + dn)^2` on the EPW energy density, not the EPW collisions; includes the kinetic half; the state's `gamma_L` when HPE or `terms.qle.landau_evolution` evolves it; signed with `terms.hpe.allow_growth` and with the QLE rate, negative = net transfer from the electrons to the EPW) | absorbed fraction / hot-electron source |
| `epw_boundary_loss` | total EPW energy removed by the absorbing boundaries per ps | — |
| `incident_flux`, `transmitted_flux` | pump flux at probes near each edge | `incident_t`, `T_t` |
| `reflected_flux`, `backrefl_flux` | Raman flux leaving left / right | `R_t` |
| `reflectivity`, `e1_sq` | legacy probes (unchanged) | — |

`post_process` logs scalar metrics with the same names and definitions as `osiris_lpi` (`laser_reflectivity`, `laser_transmissivity`, `laser_absorbed_frac`, per-quarter `_seg{i}of4` variants, `epw_growth_rate` per $\omega_0$ with the identical automated fit window, `electron_energy_frac_final`, and the `laser_incident_flux_ratio` health check). The laser budget is oriented on the pump's entrance face: x-min, or x-max when every beam is leftward; with beams entering from both faces the fraction metrics are not logged and `laser_net_flux_left` / `laser_net_flux_right` (signed +x net fluxes in units of the nominal incident flux) are.

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

With `terms.light.pump_depletion: false`, the laser is prescribed forcing and is therefore appropriate
only while depletion is negligible. With depletion enabled, the injected pump propagates through the
box and receives the reciprocal TPD and/or SRS feedback described above.

| Term | Role |
|---|---|
| $E_0$ (laser driver) | A sum of $N_c$ plane-wave components, each with amplitude $A_j$, frequency offset $\Delta\omega_j$, and phase $\phi_j$. Bandwidth models (SSD, CPP speckle) are built by populating these components — see the example decks `tpd-static-cpp-speckle.yaml` and `tpd-dynamic-ssd-speckle.yaml`. |
| $S_\text{TPD}$ | The two-plasmon-decay source coupling $E_0$ to the plasma-wave envelope. |
| $S_h$ | A noise/seed source for the plasma waves. |
| $\nu_e^\circ$ | Landau damping of the plasma-wave envelope. |

### Ion-acoustic waves

`terms.iaw.active: true` evolves the fractional ion-density perturbation and ion-velocity divergence using the MATLAB LPSE split step. The pressure contains acoustic restoring force plus ponderomotive drive from the EPW, pump, and Raman fields. The resulting density perturbation feeds back into the detuning of all three envelope equations. IAW boundary damping defaults to the EPW boundary configuration; collisional and Landau damping are independently configurable.

### Hybrid particles and the 2D boundary

The HPE particle tracker supplies self-consistent Landau-damping feedback and hot-electron diagnostics in both quasi-1D and 2D runs. For `ny > 1`, a single box-wide ensemble carries `(x, y, p_x, p_y)` and samples both electric-field components; angle-resolved projected-velocity histograms provide the resonant distribution for every `(k_x, k_y)` mode without assigning particles to individual grid cells. Periodic boundaries wrap particles, while reservoir walls use flux-weighted thermal-tail reinjection. HPE can be combined with TPD, SRS, pump depletion, and IAWs, as demonstrated by `configs/envelope-2d/tpd-srs-iaw.yaml`; `srs-hpe.yaml` remains the smaller quasi-1D example.

## What Gets Saved

**`binary/`**:

| File | Contents |
|---|---|
| `fields.xr` | The real-space envelope fields over time |
| `k-fields.xr` | The same in wavenumber space — this is where TPD growth is measured |
| `series.xr` | Scalar time series |

When IAWs are active, `fields.xr` and `k-fields.xr` also contain `iaw_density` and `iaw_velocity_divergence`; `series.xr` includes `iaw_density_sq` and `iaw_density_abs_max`.

**`plots/`**, per field `k`:

| File | Contents |
|---|---|
| `<k>_x.png`, `<k>_x_r.png` | Spatial profiles, magnitude and real part |
| `<k>.png`, `log-<k>.png`, `real-<k>.png` | Slices, linear / log / real part |
| `spacetime-<k>.png`, `spacetime-log-<k>.png`, `spacetime-real-<k>.png` | Space-time plots |

The log-scaled space-time plots are the ones to read for instability growth rates.

## Configuration Reference

See the [Configuration Reference](config.md) for complete YAML schema documentation.

## Departures from LPSE

The LPSE-parity solver paths follow the original LPSE C++ code (`original-lpse/src`) unless the
configuration reference says otherwise. The intentional departures known as of 2026-10-21, each
also noted in the configuration-reference row of the option it concerns:

| Area | LPSE | adept | Decided by / commit |
|------|------|-------|---------------------|
| HPE rate limits with `terms.hpe.energy_conservation` | clips the HPE rate (`ElectronTracker.cu:2792`) before the energy-conservation multiplier, which is applied with no second clip (`LwSolver.cpp:823`, `LightSolver.cpp:3707`); the applied rate can exceed `gamma_limit_damping` / `-gamma_limit_growth` | clips the resonant modes again after the multiplier; the multiplier's energy balance then holds only where the limits do not bind | user decision (PR #372 review), 971f999 |
| FD IAW fine-to-coarse transfer (`terms.iaw.solver: fd`, `super_samples > 1`) | point-samples the fine grid (`SuperSampleGrid_1f1::copyFineToCoarse`) and never masks the density (`isAntiAliasNelf = false`, `IawSolver.cpp:887`) | FFT truncation to the EPW grid, multiplied by the coarse IAW band (k = 0 kept) | user decision (PR #378 review), 1d2bac5 (the FFT truncation predates it) |
| FD combined Landau step (`terms.epw.solver: combined`, `terms.light.solver: fd`) | linear coefficient `gamma dt` whenever `gamma dt < 1e-3`, i.e. for every negative rate (`LwSolver.cpp:825`) | linear only for `abs(gamma dt) < 1e-3`, `-expm1(-gamma dt)` otherwise; differs only for negative rates (QLE `landau_evolution`, HPE `allow_growth`) | PR #378 / #387 review, 85f29cc |
| FD IAW noise support | the disc `abs(k) < (1 - antiAliasing.range) pi / h_coarse`, `abs(k) < iaw.maxWavenumber`, `k != 0` (`IawSolver::addNoise`, `IawSolver.cpp:2580`) | the same disc intersected with the retained band (anti-aliasing mask and `max_wavenumber`); differs only where the disc reaches past the anti-aliasing mask (e.g. `grid.dealias: rectangular` with `dy > dx`) | PR #378 review, 85f29cc |
| `terms.epw.source_window` with the combined solver | the combined path applies no `restrictRange` (`LightSolver::calculateSources`, `LwSolver::make_E0E1`) | the window multiplies the combined unified source, with either light solver | adept extension; fd path adec784 |
| `terms.epw.source.srs_k_filter` with the combined solver | `LwSolver::make_E0E1` filters only the `E0 . E1*` product, `E0` and `E1` each at its own cutoff | filters `E0` in both terms of the unified source, `E1` unfiltered | pre-existing; recorded in adec784 |
| Quasi-1D resonance-absorption k filter (`ny = 1`) | the zero-width transition makes the filter all-zero and erases the pump every damping update | a sharp passband `abs(k) <= filter_width K_nyq` | PR #378 review, 8b1430b |

The lpse2d branch with the MATLAB-prototype conventions (`lpse2d/lpse-parity`, PR #372) differs from
LPSE by design and is not covered by this table.
