# M0 — WarpX diag/runtime primitives, verified @ dev 72280884a

*2026-08-14. Every claim below was checked against the WarpX source tree at the pinned
commit (dev @ 72280884a — the Perlmutter build), not the docs site. This is the M0
record from the plan (`warpx-wrapper-plan.md` §3/§5); paths are repo-relative to the
WarpX tree.*

## Headline corrections vs the plan's assumptions

1. **PML aborts at runtime in 1D Cartesian — and PML is the DEFAULT field BC.** All
   seven damping kernels in `Source/BoundaryConditions/WarpX_PML_kernels.H` are
   `amrex::Abort("PML not implemented in 1D geometry")` under `AMREX_SPACEDIM == 1`,
   and there is no input-time guard. Every 1D deck must set `boundary.field_lo/hi`
   explicitly; the 1D absorber is **`absorbing_silver_mueller`** (Yee/FDTD only,
   which is what we run anyway). `damped` is PSATD-only.
2. **`momentum_distribution_type = maxwell_boltzmann` does not exist** (hard abort
   with rename message). Use **`maxwellian`** with `ux_std`/`uy_std`/`uz_std`
   (= sqrt(kT/mc²), dimensionless); spatially varying via
   `maxwellian_u_std_distribution_type = parser` + `ux_std_function(x,y,z)`.
   `theta` belongs to `maxwell_juttner` only. No eV inputs anywhere — convert via
   `my_constants` (`sqrt(Te_eV*q_e/(m_e*clight**2))`).
3. **ParticleHistogram2D parameter names are `_abs`/`_ord`**, not abscissa1/2, and its
   output is **openPMD meshes** (not txt, not a particle record).
4. **`gamma` is not a parser variable** in histogram/filter/value functions — use
   `sqrt(1+ux*ux+uy*uy+uz*uz)`. (`ux` etc. are γv/c.)

## Reduced diagnostics (Q1–Q5)

- Syntax: `warpx.reduced_diags_names = <n1> <n2> ...` + `<name>.type = <Type>`.
  Type strings are **case-sensitive**; all 21 at this commit include `FieldEnergy`,
  `FieldMaximum`, `FieldMomentum`, **`FieldPoyntingFlux`** (exists!), `FieldProbe`,
  `FieldReduction`, `ParticleEnergy`, `ParticleExtrema`, `ParticleHistogram`,
  `ParticleHistogram2D`, `ParticleMomentum`, `ParticleNumber`, `RhoMaximum`,
  `Timestep`. (`MultiReducedDiags.cpp`.)
- Output: `./diags/reducedfiles/<name>.txt` by default. **`<name>.path` must end in
  `/`** (string concatenation, not a join). Columns: `[0]step()`, `[1]time(s)`, then
  diag-specific; header written once with `#` prefix — **except FieldProbe, whose
  header has no `#`**. Separator single space; `precision` default 14.
- Cadence: `<name>.intervals` (default `"1"`, same slice syntax as full diags);
  global fallback `reduced_diags.intervals`. `frequency` is a hard abort (renamed).
  Iteration 0 is always written at init.
- **1D unit caveat: cell "volume" is dz** (`CellSize` returns {1,1,dz}), so
  FieldEnergy/ParticleEnergy are J per m² of transverse area. Fine for parity — the
  OSIRIS metrics are per-area in 1D too — but don't label them J.
- `FieldPoyntingFlux` in 1D emits 4 columns: `outward_power_lo_z(W)`,
  `outward_power_hi_z(W)`, `integrated_energy_loss_lo_z(J)`, `..._hi_z(J)` — i.e. it
  time-integrates for us. Step-cadence R(t)/T(t) source confirmed.
- `FieldReduction`: key is the literal 12-arg
  `<name>.reduced_function(x,y,z,Ex,Ey,Ez,Bx,By,Bz,jx,jy,jz)` (9-arg form aborts);
  `reduction_type` ∈ Maximum/Minimum/Sum/**Integral** (case-insensitive; Integral is
  an alias of Sum, both multiply by cell volume → ∫Ex² dz works as planned). Fields
  are interpolated to cell centers; requires `amr.max_level = 0`. Quote multi-token
  expressions — ParmParse concatenates tokens with **no separator**.
- `ParticleHistogram` (1D): `histogram_function(t,x,y,z,ux,uy,uz)` — **no `w`**;
  `normalization` ∈ `unity_particle_weight|max_to_unity|area_to_unity`; txt output,
  one column per bin, header carries bin centers.
- `ParticleHistogram2D`: `bin_number_abs/ord`, `bin_min/max_abs/ord`,
  `histogram_function_abs/ord(t,x,y,z,ux,uy,uz,w)`, `filter_function(...)`,
  `value_function(...)` — **always set `value_function` explicitly** (`"w"` or the
  KE·vx expression); it is invoked unconditionally and every in-tree example sets it.
  Output: openPMD dir `diags/reducedfiles/<name>/openpmd_%06T.h5`,
  `iterations[step+1].meshes["data"]`, dataset shape `(bin_number_ord,
  bin_number_abs)`, attrs `function_abscissa/ordinate`, `gridSpacing`,
  `gridGlobalOffset`. Works in 1D (CI: `inputs_test_1d_particle_absorbing_boundary`).
- `FieldProbe`: `probe_geometry = Point|Line` in 1D (`Plane` aborts); 1D reads only
  `z_probe`/`z1_probe`/`resolution` (x/y ignored). Records x,y,z,Ex..Bz,S per point;
  **one row per probe point per step**. `interp_order ≤ algo.particle_shape`.
  `integrate = 1` switches to time-integrated units. 1D-verified in
  `Examples/Physics_applications/laser_acceleration/inputs_base_1d`.

## Full diagnostics / openPMD (Q10)

- `diag1.format = openpmd`; `openpmd_backend ∈ bp5|bp4|bp|h5|json` (default prefers
  bp5 → h5 — **pin `h5` explicitly**); `openpmd_encoding ∈ f (default)|g|v`.
- Layout: `diags/<diag>/openpmd_%06T.h5` (`file_prefix`, `file_min_digits`).
  Fields split as `E/x`, `B/y`, `j/z`; scalars `rho`, `rho_<species>` are SCALAR. 1D:
  `axisLabels = {"z"}`. Particles: `position/z` only in 1D, `weighting`,
  `momentum/{x,y,z}` **converted to SI kg·m/s** (not internal γv), `id`.
- `intervals` slices `start:stop:period`, comma-separated, stop inclusive;
  `dump_last_timestep` defaults to 1.
- `fields_to_plot` is the correct param name. **Setting `diag_lo`/`diag_hi` disables
  particle output for that diag** — don't use them on a diag that also writes
  particles.

## BoundaryScraping (Q6)

Full-diag entry: `diag_type = BoundaryScraping`, `format = openpmd` (required), plus
per-species `<species>.save_particles_at_zlo/zhi = 1` (only z* exist in 1D). Output
`diags/<diag>/particles_at_zlo/openpmd_%06T.h5` with `stepScraped`,
`deltaTimeScraped`, `timeScraped` extras. `intervals` defaults to `"0"` = single dump
at the end; set it for long runs. 1D CI-covered.

## Runner-relevant (Q7–Q9)

- `geometry.dims` is compile-time-checked: a 1D run needs the 1D binary; mismatch is
  an explicit abort. CMake produces `warpx.1d.MPI.CUDA.DP.PDP...` **plus the stable
  symlink `build/bin/warpx.1d`** — point `WARPX_BIN_1D` at the symlink.
- CLI: `warpx <inputs> [key=value ...]`, arrays as separate unquoted tokens. All argv
  handling is AMReX's; no `--` separator.
- `warpx_used_inputs` is written **by default** (param `warpx.used_inputs_file`).
- Error text: WarpX aborts/asserts print `### ERROR   : ` (ablastr TextMsg);
  warnings `!!! WARNING : `; a `[ THE END ]` warning summary is printed
  unconditionally at the end. There is **no** "simulation completed" string; the
  `Total Time : ` line is gated on `warpx.verbose` (default 1). AMReX itself is not
  vendored, so its abort banner/exit codes were not verifiable — the runner treats
  the exit code as primary and the tokens as detail only.
- **Exit-0 failure modes to remember**: (a) typo'd/unknown input params are silently
  ignored (`amrex.abort_on_unused_inputs` default 0 — and it aborts only *after*
  completion when set); (b) warnings never abort unless
  `warpx.abort_on_warning_threshold` is set; (c) `warpx.break_signals` ends the run
  early with exit 0 — compare the last `STEP <n> ends.` against the plan. M2's
  postproc should cross-check `final_step` vs `max_step`/`stop_time`.

## Laser + species (Q11–Q13)

- `<laser>.profile ∈ gaussian|parse_field_function|from_file` (lowercased on read;
  `from_file` **aborts in 1D**). Required always: `position` (3 components),
  `direction` (must be `0 0 ±1` in 1D), `polarization` (3, ⟂ direction),
  `wavelength`, and **exactly one of `e_max` / `a0`** (XOR-asserted — the wrapper's
  write_units handles both). Gaussian additionally requires `profile_waist`,
  `profile_duration`, `profile_t_peak`, `profile_focal_distance` (waist has no 1D
  effect but is still mandatory). A laser positioned outside the domain is
  **silently disabled** (warning only).
- **Laser-off gap / custom envelope: `profile = parse_field_function` with
  `<laser>.field_function(X,Y,t)`** (capitals; full field incl. carrier, V/m);
  `e_max|a0` + `wavelength` still required (they set antenna weight). In-tree 1D
  template with ramp envelope: `inputs_test_1d_particle_absorbing_boundary`
  (`Emax*cos(omega0*t)*if(...)` — prefix `if(t<t_delay, 0, ...)` for the gap).
- Density: `<species>.profile = parse_density_function` +
  `density_function(x,y,z)` confirmed; optional `density_min/max`; region limits
  `zmin/zmax`. ppc: `injection_style = NUniformPerCell` with
  `num_particles_per_cell_each_dim` = exactly **one** entry in 1D.
- Particle BCs: `boundary.particle_lo/hi = absorbing` (default) confirmed; `thermal`
  needs `boundary.<species>.u_th`.

## Best in-tree templates

- 1D laser + parse_density_function + ParticleHistogram2D + Silver-Mueller:
  `Examples/Tests/particle_absorbing_boundary/inputs_test_1d_particle_absorbing_boundary`
- 1D FieldProbe Line (+ moving window): `Examples/Physics_applications/laser_acceleration/inputs_base_1d`
- 1D openPMD + BoundaryScraping: `Examples/Tests/ionization_dsmc/inputs_test_1d_photoneutralization_dsmc`
- Every reduced-diag's exact syntax: `Examples/Tests/reduced_diags/inputs_test_3d_reduced_diags`
