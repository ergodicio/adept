# Normalization Fix Implementation Plan

Reference: `docs/normalization_derivation.pdf`

## Summary of Bugs

Three normalization bugs in the Vlasov-1D solver:

1. **Ponderomotive force sign error + species-independence** (`field.py:346`): The code computes `pond = -0.5 * grad(a^2)` which is the electron ponderomotive *acceleration*, then passes it as an electric field to `edfdv`, which multiplies by `q/m` again. For electrons this flips the sign (pushes toward high intensity). For multi-species, the ponderomotive E-field is species-dependent.

2. **`ex_driver` wrong prefactor** (`field.py:23`): Uses `|k|` instead of `ω`. Missing factor of `c_norm = c/v₀`.

3. **`ey_driver` wrong prefactor and sign** (`field.py:23`): Uses `+|k|` instead of `-ω²`.

## Change 1: Split `Driver` into `ExDriver` and `EyDriver`

### File: `adept/_vlasov1d/solvers/pushers/field.py`

Replace the current `Driver` class (lines 11–31) with two classes:

```python
class LongitudinalElectricFieldDriver:
    """Electric field driver for the Vlasov equation.

    Computes the normalized electric field E_tilde = ω * a0 * sin(kx - ωt)
    from a specified EM wave. This is added directly to the electrostatic
    field in the Vlasov push.
    """
    def __init__(self, xax, drivers: list[EMDriver]):
        self.xax = xax
        self.drivers = drivers

    def _single_driver_field(self, driver: EMDriver, current_time):
        kk = driver.k0
        ww = driver.w0
        dw = driver.dw0
        factor = driver.envelope(self.xax, current_time)
        # E = -dA/dt → normalized: E_tilde = ω * a0 * sin(kx - ωt)
        return factor * (ww + dw) * driver.a0 * jnp.sin(kk * self.xax - (ww + dw) * current_time)

    def __call__(self, t, args):
        total_de = jnp.zeros_like(self.xax)
        for pulse in self.drivers:
            total_de += self._single_driver_field(pulse, t)
        return total_de


class TransverseCurrentSourceDriver:
    """Transverse current source for the wave equation.

    Computes the normalized source S_tilde = -ω² * a0 * sin(kx - ωt)
    for injection into the wave equation ∂²a/∂t² = c²∂²a/∂x² - (ne/n0)a + S.
    """
    def __init__(self, xax, drivers: list[EMDriver]):
        self.xax = xax
        self.drivers = drivers

    def _single_driver_source(self, driver: EMDriver, current_time):
        kk = driver.k0
        ww = driver.w0
        dw = driver.dw0
        factor = driver.envelope(self.xax, current_time)
        # Source = -ω² * a0 * sin(kx - ωt) (the ∂²ₜa contribution)
        return -factor * (ww + dw)**2 * driver.a0 * jnp.sin(kk * self.xax - (ww + dw) * current_time)

    def __call__(self, t, args):
        total = jnp.zeros_like(self.xax)
        for pulse in self.drivers:
            total += self._single_driver_source(pulse, t)
        return total
```

### File: `adept/_vlasov1d/solvers/vector_field.py`

Update `VlasovMaxwell.__init__` (line 285–286) to use the new classes:

```python
# Before:
self.ey_driver = field.Driver(grid.x_a, drivers=drivers.ey)
self.ex_driver = field.Driver(grid.x, drivers=drivers.ex)

# After:
self.ey_driver = field.TransverseCurrentSourceDriver(grid.x_a, drivers=drivers.ey)
self.ex_driver = field.LongitudinalElectricFieldDriver(grid.x, drivers=drivers.ex)
```

No other changes needed here — the call sites (`self.ex_driver(t, args)` and `self.ey_driver(t, args)`) have the same interface.

## Change 2: Ponderomotive Force — Pass as Acceleration, Not E-field

The ponderomotive acceleration is species-dependent: `a_pond_s = -(q̃²)/(2m̃²) ∂(ã²)/∂x̃`. Rather than pretending it's an E-field and bundling it with the Poisson field, we pass `grad_a2 = ∂(ã²)/∂x̃` separately to `edfdv`, which applies the correct species-dependent coefficient.

### File: `adept/_vlasov1d/solvers/pushers/field.py`

Change `ElectricFieldSolver.__call__` (lines 330–348) to return `grad_a2` instead of the ponderomotive "force":

```python
def __call__(self, f_dict, a, prev_ex, dt):
    # Return the raw gradient of a^2 (without any coefficient).
    # The species-dependent ponderomotive coefficient is applied in the edfdv push.
    grad_a2 = jnp.gradient(a**2.0, self.dx)[1:-1]
    self_consistent_ex = self.es_field_solver(f_dict, prev_ex, dt)
    return grad_a2, self_consistent_ex
```

### File: `adept/_vlasov1d/solvers/pushers/vlasov.py`

FIXME: the argument to edfdv should just be called `force`, and it should be divided by particle mass to get acceleration. This puts too much logic about the ponderomotive force in thes classes.
Add a `grad_a2` argument to the `push` method of both `VelocityExponential` and `VelocityCubicSpline`. The ponderomotive contribution to the velocity shift is `-(q̃²)/(2m̃²) * grad_a2 * dt`, which does NOT go through the `qm * E` pathway.

**`VelocityExponential.push`** (currently lines 63–71):

```python
def push(self, f_dict, e, grad_a2, dt):
    result = {}
    for species_name, f in f_dict.items():
        kv_real = self.species_grids[species_name]["kvr"]
        qm = self.species_params[species_name]["charge_to_mass"]
        q = self.species_params[species_name]["charge"]
        m = self.species_params[species_name]["mass"]
        # Electrostatic acceleration: (q/m) * E
        # Ponderomotive acceleration: -(q^2)/(2*m^2) * grad(a^2)
        accel = qm * e - (q**2 / (2.0 * m**2)) * grad_a2
        result[species_name] = jnp.real(
            jnp.fft.irfft(
                jnp.exp(-1j * kv_real[None, :] * dt * accel[:, None])
                * jnp.fft.rfft(f, axis=1),
                axis=1,
            )
        )
    return result
```

**`VelocityExponential.__call__`** — update signature to pass `grad_a2` through:

```python
def __call__(self, f_dict, e, grad_a2, dt):
    if self.parallel:
        return shard_map(
            self.push, mesh=self.mesh,
            in_specs=(P("device", None), P("device"), P("device"), P()),
            out_specs=P("device", None)
        )(f_dict, e, grad_a2, dt)
    else:
        return self.push(f_dict, e, grad_a2, dt)
```

**`VelocityCubicSpline.push`** (currently lines 91–100) — same pattern:

```python
def push(self, f_dict, e, grad_a2, dt):
    result = {}
    for species_name, f in f_dict.items():
        v = self.species_grids[species_name]["v"]
        qm = self.species_params[species_name]["charge_to_mass"]
        q = self.species_params[species_name]["charge"]
        m = self.species_params[species_name]["mass"]
        nx = f.shape[0]
        v_repeated = jnp.repeat(v[None, :], repeats=nx, axis=0)
        # Total acceleration = (q/m)*E - (q^2)/(2m^2) * grad(a^2)
        accel = qm * e - (q**2 / (2.0 * m**2)) * grad_a2
        vq = v_repeated - accel[:, None] * dt
        result[species_name] = self.interp(xq=vq, x=v_repeated, f=f)
    return result
```

**`VelocityCubicSpline.__call__`** — same signature update as `VelocityExponential.__call__`.

### File: `adept/_vlasov1d/solvers/vector_field.py`

FIXME: these should still be called `pond`, it'll just have units of force. In the field_solve function we should perform the constant factor multiplication from above.
Update all call sites in the integrators to pass `grad_a2` separately instead of adding `pond` to `e`.

**`LeapfrogIntegrator.__call__`** (lines 82–90):

```python
def __call__(self, f_dict, a, dex_array, prev_ex):
    f_after_v = self.vdfdx(f_dict, dt=self.dt)
    if self.field_solve.hampere:
        f_for_field = f_dict
    else:
        f_for_field = f_after_v
    grad_a2, e = self.field_solve(f_dict=f_for_field, a=a, prev_ex=prev_ex, dt=self.dt)
    f_dict = self.edfdv(f_after_v, e=e + dex_array[0], grad_a2=grad_a2, dt=self.dt)
    return e, f_dict
```

**`SixthOrderHamIntegrator.__call__`** (lines 140–186): Same pattern at each of the 6 substeps. At each substep, replace:

```python
# Before:
ponderomotive_force, self_consistent_ex = self.field_solve(...)
force = ponderomotive_force + dex_array[i] + self_consistent_ex
f_dict = self.edfdv(f_dict, e=force, dt=...)

# After:
grad_a2, self_consistent_ex = self.field_solve(...)
f_dict = self.edfdv(f_dict, e=dex_array[i] + self_consistent_ex, grad_a2=grad_a2, dt=...)
```

This pattern repeats 6 times in the method (for substeps 0–5).

### File: `adept/_vlasov1d/solvers/pushers/vlasov.py` (class `VlasovExternalE`)

This class (lines 14–52) has its own `step_edfdv` method that doesn't use the standard `edfdv` interface. It doesn't involve ponderomotive forces, so it does not need changes for the ponderomotive fix. However, verify it doesn't interact with the modified interfaces.

## Change 3: Update Documentation

### File: `docs/source/solvers/vlasov1d/config.md`

Update the drivers section to document:
- `ex` drivers produce normalized electric field with prefactor `ω * a0`
- `ey` drivers produce wave equation source with prefactor `-ω² * a0`
FIXME: so we're on the same page, a0 _means the same thing_ for both driver types. They just vary in how a0 is used to determine the magnitude of some forcing term -- they produce different forcing terms which enter the respective (vlasov or wave) equations differently. But in both cases a0 means the same normalized vector potential of the wave.
- The meaning of `a0` for each driver type

## Test Plan

### Test 1: `ex_driver` Quiver Velocity (`test_ex_driver_quiver.py`)

**Physics:** An electron in a uniform oscillating electric field `E = E₀ sin(ωt)` acquires a quiver velocity `v_quiver = (eE₀)/(mₑω) cos(ωt)`. In normalized units, for a driver with amplitude `ã₀` and frequency `ω̃`:

- The normalized E-field is `Ẽ = ω̃ · ã₀`
- The quiver velocity amplitude is `ṽ_quiver = ã₀` (the `ω̃` in E cancels with the `1/ω̃` from integration)

**Setup:**
- Uniform plasma, very low density (e.g., `n₀ → 0` or just use ion background to neutralize) so Poisson self-consistent fields are negligible
- Single `ex` driver with known `a0`, `k0 ≈ 0` (long wavelength, spatially uniform across the box), and `w0` set to some reasonable value
- Turn off Fokker-Planck collisions
- No `ey` drivers (no ponderomotive force)
- Run for several oscillation periods

**Measurement:**
- Extract the mean electron velocity `<v> = ∫v·f dv / ∫f dv` at each timestep
- Fit the oscillation amplitude and compare with the expected `ṽ_quiver = ã₀`
- Tolerance: 5% (accounting for finite grid effects)

**Config sketch:**
```yaml
units:
  normalizing_temperature: 2000eV
  normalizing_density: 1e18/cc    # Very low density to minimize self-consistent response
grid:
  nx: 32
  # FIXME: I bet we can go lower with nv, like 128
  nv: 512
  vmax: 6.0
  xmin: 0.0
  xmax: 20.94      # = 2π/k0, one wavelength
  dt: 0.1
  tmax: 200.0      # Several oscillation periods at ω ~ 1
density:
  species-background:
    basis: uniform
    # ... uniform n=1
drivers:
  ex:
    '0':
      params:
        a0: 0.001    # Small amplitude (linear regime)
        k0: 0.3      # k0 = 2π/xmax
        w0: 1.0      # Arbitrary frequency
        dw0: 0.0
      envelope:
        time: { center: 1000, rise: 5.0, width: 2000 }
        space: { center: 0.0, rise: 10.0, width: 1e6 }
  ey: {}
terms:
  field: poisson
  edfdv: exponential
  time: leapfrog
  fokker_planck: { is_on: false }
  krook: { is_on: false }
```

**Assertion:**
```python
# Measured quiver velocity amplitude from time series of <v>(t)
# Expected: v_quiver = a0 (in normalized units)
assert abs(measured_amplitude - a0) / a0 < 0.05
```

### Test 2: Ponderomotive Force from Beat Wave (`test_ey_ponderomotive_quiver.py`)

**Physics:** Two co-propagating EM waves with slightly different frequencies create a beat pattern. The slowly-varying ponderomotive force from the beat drives a low-frequency electron oscillation.

Given two `ey` drivers with frequencies `ω₁`, `ω₂` (and corresponding wavenumbers `k₁`, `k₂`), the beat wave has:
- Beat frequency: `Δω = ω₁ - ω₂`
- Beat wavenumber: `Δk = k₁ - k₂`
- Ponderomotive potential: `∝ a₁ a₂ cos(Δk·x - Δω·t)`

The ponderomotive force on electrons is `F_pond = -∂ₓ U_pond`. The resulting quiver velocity in the beat pattern:

For two waves with equal amplitude `ã₀`, the slowly-varying part of `ã²` is `2ã₀² cos(Δk·x - Δω·t)` (from the cross term; the self-terms are at 2ω which average out). The ponderomotive acceleration amplitude is:

```
a_pond = (1/2) * Δk * 2 * ã₀² = Δk * ã₀²
```

The velocity oscillation amplitude at the beat frequency is:
```
ṽ_pond = a_pond / Δω = Δk * ã₀² / Δω
```

**Setup:**
- Uniform plasma with density such that `ω₁`, `ω₂` are well above `ωpe` (wave propagates freely)
- Two `ey` drivers with frequencies `ω₁ = ω_laser`, `ω₂ = ω_laser - Δω` where `Δω` is small
- Wavenumbers from the dispersion relation: `k = sqrt(ω² - ωpe²)/c`
- Turn off Fokker-Planck collisions
- No `ex` drivers
- Run long enough for the beat oscillation to develop (several periods of `2π/Δω`)

**Measurement:**
- Extract the mean electron velocity `<v>(x, t)` from the distribution function
- Fourier transform in space and time to isolate the `(Δk, Δω)` component
- Measure its amplitude and compare with `Δk · ã₀² / Δω`
- This tests: (a) the `ey_driver` launches waves with the correct amplitude, (b) the ponderomotive force has the correct sign and magnitude

**Config sketch:**
```yaml
units:
  normalizing_temperature: 2000eV
  normalizing_density: 1e20/cc    # ωpe/ωp0 = 1, moderate density
grid:
  nx: 512
  nv: 256
  vmax: 6.0
  xmin: 0.0
  xmax: <computed>   # Several beat wavelengths 2π/Δk
  dt: <computed>     # Overridden for EM stability
  tmax: <computed>   # Several beat periods 2π/Δω
density:
  species-background:
    basis: uniform
    # n = 1 (uniform)
drivers:
  ex: {}
  ey:
    '0':
      params:
        intensity: <computed>   # Gives a0 ~ 0.01 (small but measurable)
        wavelength: <computed>  # ω₁ from dispersion
      envelope:
        time: { center: <large>, rise: <gradual>, width: <large> }
        space: { center: <center>, rise: <wide>, width: <wide> }
    '1':
      params:
        intensity: <computed>   # Same intensity as driver 0
        wavelength: <computed>  # ω₂ = ω₁ - Δω
      envelope:
        # Same as driver 0
terms:
  field: poisson
  edfdv: exponential
  time: leapfrog
  fokker_planck: { is_on: false }
  krook: { is_on: false }
```

**Assertion:**
```python
# Extract (Δk, Δω) Fourier component of mean velocity
# Expected amplitude: Δk * a0^2 / Δω
measured = abs(fft_component_at_beat)
expected = delta_k * a0**2 / delta_omega
assert abs(measured - expected) / expected < 0.15  # 15% tolerance (numerical diffusion, finite grid)
```

**Implementation notes for this test:**
- This test is more involved than Test 1. The key challenge is choosing parameters where: (a) both waves propagate cleanly, (b) the beat frequency is well-resolved by the time grid, (c) `a0` is small enough to stay linear but large enough for the ponderomotive signal to be above numerical noise.
- A good starting point: `ω₁ ≈ 2ωpe`, `Δω ≈ 0.1ωpe`, which gives a beat period of `~60 ωpe⁻¹`.
- The ponderomotive velocity will be of order `a0² * Δk / Δω`, so for `a0 ~ 0.01` and `Δk/Δω ~ O(1)`, the signal is `~1e-4` in normalized velocity—small but detectable with sufficient resolution.

### Test 3: Regression — Re-run `test_em_dispersion.py`

The existing dispersion test (`tests/test_vlasov1d/test_em_dispersion.py`) verifies that EM waves propagate with the correct dispersion relation. After the `EyDriver` fix, the source amplitude changes dramatically (by a factor of `~c_norm² |k|`), so the test config's `intensity` parameter may need adjustment to get a wave amplitude in the right range. Re-run this test and adjust the driver intensity if needed to keep the wave in the linear regime.

### Test 4: Regression — Re-run `test_landau_damping.py`

The Landau damping test uses `ex_driver` with the `a0/k0/w0` parametrization. After the `ExDriver` fix, the driver amplitude changes by a factor of `ω/|k|`. The test config uses `a0: 1e-6` which is in the linear regime; verify the damping rate and frequency still match theory. The `a0` value may need to be adjusted by a factor of `|k|/ω` to maintain the same physical amplitude.

## Execution Order

Use red-green TDD:
1. **Tests** — write Test 1 and Test 2 first, then run them (they should fail on the current code and pass after the fixes). Then verify regression tests still pass.
2. **Change 1** (Driver split) — can be done independently, only touches `field.py` and `vector_field.py`
3. **Change 2** (Ponderomotive force) — larger change touching `field.py`, `vlasov.py`, and `vector_field.py`
4. **Change 3** (Docs)

## Files Modified (Summary)

| File | Changes |
|------|---------|
| `adept/_vlasov1d/solvers/pushers/field.py` | Replace `Driver` with `ExDriver`+`EyDriver`; change `ElectricFieldSolver` to return `grad_a2` |
| `adept/_vlasov1d/solvers/pushers/vlasov.py` | Add `grad_a2` arg to `VelocityExponential` and `VelocityCubicSpline` |
| `adept/_vlasov1d/solvers/vector_field.py` | Update `VlasovMaxwell` to use new driver classes; update `LeapfrogIntegrator` and `SixthOrderHamIntegrator` to pass `grad_a2` |
| `docs/source/solvers/vlasov1d/config.md` | Document driver normalization |
| `tests/test_vlasov1d/test_ex_driver_quiver.py` | New test |
| `tests/test_vlasov1d/test_ey_ponderomotive_quiver.py` | New test |
