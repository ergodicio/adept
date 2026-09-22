# MAGPIE reference scales and flow analysis

VFP-2D can analyze a prescribed reconnection region using ion-flow, magnetic,
and electron-distribution diagnostics. The intended first comparison is the
carbon MAGPIE platform. A reduced calculation can test transport, energy
partition and sensitivity to imposed upstream conditions alongside GORGON.
It does not yet predict wire ablation, the pulsed-power circuit, radiative
transfer, evolving charge states, or a validated experiment outcome.

## A reproducible carbon reference

The defaults in `adept.vfp2d.analysis.MagpieReference` are upstream values from
[Hare et al., Physical Review Letters 118, 085001 (2017), Table I](https://arxiv.org/pdf/1609.09234):

| Quantity | Upstream reference |
| --- | ---: |
| Electron density | $3\times10^{17}\ \mathrm{cm}^{-3}$ |
| Mean ion charge | 4 |
| Electron temperature | 15 eV |
| Ion temperature | 50 eV (the reported upstream upper bound) |
| Reconnecting magnetic field | 3 T |
| Inflow speed | 50 km/s |
| Harris sheet half-width $\delta$ | 0.6 mm |

The same paper reports sheet values $n_e=6\times10^{17}$ cm$^{-3}$,
$\bar Z=6$, $T_e=100$ eV, $T_i=600$ eV, and an outflow of 130 km/s.
Those are comparison targets, not imposed upstream conditions. The measured
heating exceeds the classical dissipation estimate; matching input scales
does not establish that a fluid-ion model can reproduce it.

The characteristic sheet half-length $L=7$ mm follows the definition in
[Hare et al., Physics of Plasmas 25, 055703 (2018), section III](https://arxiv.org/pdf/1711.06534).
This is a layer scale, not the box size or wire-array radius. That study also
shows that changing electrode placement changes density, magnetic field, and
layer geometry together. A local parameter scan does not establish which
combinations are experimentally accessible; a global model or measurement
must supply that mapping.

Generate the SI scale report without launching a simulation:

```bash
python -m adept.vfp2d.analysis
```

To evaluate another upstream state, provide a JSON object with any of the
`MagpieReference` field names. All lengths, densities, fields and speeds use
SI; temperatures use eV; `ion_mass_u` uses unified atomic mass units.

```bash
python -m adept.vfp2d.analysis --parameters carbon_upstream.json
```

For example:

```json
{"electron_density_m3": 4e23, "magnetic_field_t": 3.5}
```

The reference calculation uses carbon mass $12\,u$ and produces:

| Derived quantity | Value |
| --- | ---: |
| Ion mass / electron mass | 21874.7 |
| Alfvén speed $B/\sqrt{\mu_0\rho_i}$ | 69.2 km/s |
| Alfvén Mach number | 0.722 |
| Isothermal ion-acoustic speed $\sqrt{e(\bar ZT_e+T_i)/m_i}$ | 29.7 km/s |
| Isothermal sonic Mach number | 1.68 |
| Magnetic pressure $B^2/(2\mu_0)$ | 3.58 MPa |
| Ram pressure $\rho_i u_{in}^2$ | 3.74 MPa |
| Scalar thermal pressure $e(n_eT_e+n_iT_i)$ | 1.32 MPa |
| Dynamic beta | 1.04 |
| Thermal beta | 0.369 |
| Ion inertial length $\sqrt{m_i/(\mu_0n_i\bar Z^2e^2)}$ | 0.717 mm |
| $d_i/\delta$ | 1.20 |
| $L/\delta$ | 11.7 |
| Inflow crossing time $\delta/u_{in}$ | 12 ns |
| Alfvén transit time $L/v_A$ | 101 ns |

These are recomputed from the listed inputs. For example, the thermal beta
agrees with the approximately 0.4 value in the 2018 platform table; it need not
match every rounded value in the earlier article's prose. The sonic speed
above is isothermal and must not be substituted for an adiabatic CFL wave
speed. Dynamic beta uses ram pressure, so $\beta_{dyn}=2M_A^2$; it is twice
the directed kinetic-energy-density / magnetic-pressure ratio.

No resistivity is selected by this report. In particular, the reported
$S\simeq120$ is a layer-temperature estimate, not the result of putting the
15 eV upstream temperature into a transport formula. Cooling, Coulomb mean
free paths, ion viscosity and ionization require specified closures and
local state. They are not silently inferred from these reference inputs.

## Diagnostics in `moments.nc`

The geometry convention is **x along the sheet/outflow and y across the
sheet/inflow**, interchanged relative to the experimental papers' labels.
The topology diagnostic samples a fixed center near the coordinate origin
(or box midpoint when the origin is outside the domain). It requires
opposite upstream $B_x$, reasonable two-sided balance, a small central
in-plane field, an $A_z$ saddle and a central current peak. It is not a search
for displaced X-points or multiple plasmoids. For asymmetric or migrating
sheets, inspect the full fields or supply a separate topology analysis.

The upstream locations are the two peaks of $|B_x|$ on this central x line.
Their saved coordinates, `upstream_y_lower` and `upstream_y_upper`, make the
sampling reproducible and reveal motion or switches between peaks. The
existing `normalized_reconnection_rate` retains its Nernst normalization.
A zero-Nernst run now correctly marks that normalization invalid while still
allowing the independent bulk and Alfvén diagnostics.

| Variable | Meaning |
| --- | --- |
| `upstream_ion_inflow_y` | Mean signed inward ion speed; negative contributions remain negative |
| `upstream_ion_inflow_y_lower`, `upstream_ion_inflow_y_upper` | Separate sides, each positive inward |
| `upstream_alfven_speed` | Mean of the two reconnecting-field Alfvén speeds |
| `normalized_reconnection_rate_bulk` | Central $E_z/\langle |B_x|u_{i,in}\rangle$; requires centered topology and **both** sides inward |
| `normalized_reconnection_rate_alfven` | Central $E_z/\langle |B_x|v_A\rangle$; requires centered topology and a positive finite scale |
| `bulk_rate_normalization_valid`, `alfven_rate_normalization_valid` | Independent validity masks, unrelated to the Nernst run maximum |
| `centerline_ion_outflow_speed` | Mean of each side's peak outward speed on the full center y line |
| `upstream_ram_pressure`, `upstream_magnetic_pressure`, `upstream_thermal_pressure` | Two-side mean pressure scales |
| `upstream_dynamic_beta`, `upstream_thermal_beta` | Ratios to total magnetic pressure, including guide field |
| `upstream_electric_flux_inflow` | Signed inward magnetic-flux transport from the full $E_z$ |
| `upstream_bulk_flux_inflow` | Same measurement using only $-(\mathbf u_i\times\mathbf B)_z$ |
| `current_sheet_gradient_half_width` | $\langle|B_x|\rangle/|\partial_yB_x|$ at center; Harris $\delta$ only for a resolved Harris profile |

The electric-flux diagnostic uses $\langle s\,\mathrm{sign}(B_x) E_z\rangle$,
where $s=+1$ below and $-1$ above the sheet. It separates total transport from
bulk-ion transport without pretending that $E_z/B_x$ is a unique magnetic
advection velocity. Its sign is positive for flux delivered towards the
sheet; the central signed reconnection rates follow the stored $E_z$ sign.
Neither a finite inflow speed nor a finite flux budget independently proves
reconnection.

In normalized solver units, $p_B=c_{norm}^2|B|^2/2$ and
$v_A=c_{norm}|B_x|/\sqrt{\rho_i}$. Missing `light_speed_normalized` metadata
makes these diagnostics NaN rather than assuming a normalization.
`temperature_energy_normalized` supplies $T_0/(m_ev_0^2)$ for electron
pressure. Saved temperatures and pressures must not be divided without this
factor. The saved electron temperature is a scalar second moment in the ion
frame; the reported thermal beta is a weak-drift thermal-pressure proxy and
includes any resolved relative electron-drift contribution. New MLflow rate metrics omit invalid values and retain explicit
valid-fraction metrics. The legacy Nernst metrics retain their existing
zero-on-invalid convention.

The X-point history and Ohm-term plots include the ion bulk term.
`plots/reconnection/flow_history.png` separates velocity, pressure ratios,
gated rates and flux transport. `topology_ion_final.png` overlays ion-flow
arrows; the existing Nernst topology plot remains available. The
`current_sheet_rms_width` remains a current-weighted RMS over a finite central
window; it is **not** interchangeable with the Harris half-width.
Peak outflow is an easily inspected diagnostic, not a mass-flux-weighted
measurement at an experimental observation chord.

## A fixed-line Faraday budget

The `centerline_lower_*` and `centerline_upper_*` variables integrate signed
$B_x$ from the fixed lowest saved y cell center to the sheet center, and from
there to the highest saved y cell center. For either interval $[a,b]$:

$$
\Phi=\int_a^b B_x\,dy,\qquad
\dot\Phi=E_z(a)-E_z(b).
$$

`bx_flux`, `faraday_rate`, and `faraday_residual` save the inventory, electric
flux difference and their mismatch. The raw residual includes external magnetic
drive and must not be treated as a numerical conservation error in a driven run.
The residual uses finite differences between saved times and trapezoidal spatial integration; refine the save
cadence and grid before interpreting it as an evolution error. A single
snapshot has a NaN residual. The integrals retain mean magnetic flux that
the periodic $A_z$ reconstruction excludes. They do not assume moving
upstream peaks are fixed integration boundaries and do not cover the
unsaved half-cell beyond either edge. They are local Faraday consistency
checks, not full-domain energy or mass conservation claims.

When `reservoir_magnetic_field_change(t,x,y,component)` is present, the analysis
also saves each side's `source_bx_flux` and
`source_accounted_faraday_residual`. The former integrates the cumulative
**measured** magnetic increment of the reservoir on the same fixed interval;
the latter is

$$
\frac{d}{dt}\int_a^b(B_x-\Delta B_{x,R}^{cumulative})\,dy
- [E_z(a)-E_z(b)].
$$

This subtracts the implemented source, including spatial-envelope derivatives,
without constructing an artificial source electric field. It retains errors
from time sampling, spatial quadrature and the physical update. Missing source
history leaves the original diagnostics unchanged; a single snapshot still
has no evaluable time derivative.

## What a comparison can establish

A useful sequence is to verify magnetic-pressure acceleration and energy
transfer, establish converged colliding-flow transport, and then compare
field reversal, compression, flow, width and electron/ion heating with a
specified experimental state or GORGON extraction. Keep each imposed
profile and observation chord with the result. A short periodic-box run
alone cannot establish sustained experiment-scale reconnection: boundary
reservoirs, open exhausts, drive history, box-size independence and adequate
runtime must be demonstrated separately.

The carbon reference has $d_i/\delta$ of order unity. Collisional ions can
justify a fluid moment description in some regions without removing Hall
physics or proving an isotropic scalar-pressure ion closure adequate.
Counterstreaming ion populations, ion viscosity and anomalous heating need
additional validation. Fixed $\bar Z=4$ conserves one chosen charge state;
it cannot reproduce the measured increase to 6 in the sheet. The platform
paper's carbon layer cooling time (about 600 ns) is longer than a 12 ns
crossing time, but this does not make radiation negligible everywhere or
for an entire drive. Aluminium has much stronger cooling and requires a
different validation program. Use these limitations to choose an analysis
question and comparison observables, rather than treating agreement in
one diagnostic as experimental design validation.
# Review trigger: this page defines the analysis contract for MAGPIE-scale comparisons.
