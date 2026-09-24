# FARSIGHT-1D

`farsight-1d` is an experimental, independent JAX implementation of the
**direct-sum, fixed or adaptive panel** variant of the method described by Sandberg, Krasny
and Thomas in [The FARSIGHT Vlasov-Poisson code](https://doi.org/10.1016/j.jcp.2024.113664),
JCP 523 (2025), 113664. It lives in `adept.farsight1d` and uses ADEPT's new
`SolverBuilder`, `PreparedSimulation`, and `ScanProgram` interfaces.

It solves collisionless 1D1V electron Vlasov-Poisson in normalized plasma units,
with periodic x, a finite velocity interval and a homogeneous neutralizing
background. It is a reference implementation for independent discretization
comparisons. Bounded quadtree AMR is optional; the original code's barycentric
treecode is **not implemented**. It has not been validated for production
turbulence or benchmarked on GPUs.

## Numerical method

The reference mesh has `(nx + 1, nv + 1)` nodes. Both interval counts are even;
each panel comprises a shared 3 by 3 set of nodes. The duplicated periodic x
endpoint has half weight, as do the velocity endpoints under the default
composite trapezoid rule. Composite Simpson quadrature is optional.

The field is calculated directly from the moving nodes, with no deposition
grid or field interpolation:

$$
E(x_i)=\sum_j q f_j W_j K_\epsilon(x_i-x_j),\qquad
K_\epsilon(d)=\frac{r\sqrt{1+4a^2}}{2\sqrt{r^2+a^2}}-r,
$$

where $r=\operatorname{wrap}(d/L,[-1/2,1/2))$ and $a=\epsilon/L$.
The positive softening length `epsilon` is in normalized x units. The kernel
projects out the mean charge: a uniform background has zero field, and no
net-charge mode is represented if remeshing changes the electron mass.

Each step advances $\dot x=v$, $\dot v=-E(x)$ with coupled classical RK4,
recomputing the field from all stage positions at each stage. x remains
unwrapped to preserve panel connectivity. Nodal values and material quadrature
weights are held fixed during this push.

By default every push is followed by remeshing onto the reference nodes.
Straight quadrilateral edges locate each target in a deformed panel. A 9 by 9
linear solve fits the physical-coordinate polynomial
$\sum_{i,j=0}^2 c_{ij}(x-x_c)^i(v-v_c)^j$ to that panel's nine values. Coordinates
are scaled to improve conditioning. This is not interpolation in inverse-mapped
reference coordinates. Targets outside the advected velocity domain receive
zero; choosing a sufficiently wide velocity interval remains necessary.

Shared edges use the first containing panel in deterministic index order.
The periodic duplicate endpoint reuses the first endpoint's interpolated value.
Folded, collapsed, singular, nonfinite or overly extended panels produce an
invalid result and an analyzer error. Reduce the timestep or remesh interval
when a panel no longer fits in one nearest-periodic-image neighborhood. The
current implementation does not use the original code's optional unshearing
or clipping.

## Adaptive panels

Enable `amr.enabled` for quadtree leaves with nine nodes per panel. At each
remesh the distribution is reconstructed from the deformed active leaves,
sampled on a bounded candidate hierarchy, and the leaf partition is rebuilt
from the roots. A panel refines when its nine-point range exceeds
`atol + rtol * max(abs(f))`, or it is below `min_level`. With `rtol=0` this
matches the first trigger in the author's implementation; the relative term
is an ADEPT extension. Face neighbors are balanced to a level difference of
at most one, including across the periodic x seam. The current balancing pass
conservatively considers the whole neighboring subtree, so it can refine more
than strictly face-local balancing requires. Newly created leaves are tested
again until indicator and balance decisions reach a fixed point. Rebuilding
allows coarsening.

Active leaves are packed into `max_panels` fixed-shape slots. Local trapezoid
or Simpson weights include each incident panel's contribution at duplicated
nodes, so their sum equals the domain area for every valid partition. Inactive
slots have zero f and weight and safe finite coordinates. Capacity overflow
sets persistent failure and the host analyzer raises; a truncated partition
is never reported as a successful solution. Saturation at `max_level` is a
separate diagnostic, not evidence that the requested tolerance was met.

Nonconforming straight-edge panels can leave small gaps or overlaps after a
nonlinear push, even with 2:1 balance. The finest containing leaf owns overlaps.
Interior gaps use the nearest polygon's physical biquadratic, with distance
measured in one fixed metric `(x/L, v/(vmax-vmin))`. Gap count and maximum
extension are recorded, and exceeding `max_gap_fraction` fails the solve.
This explicit polynomial extension is an **ADEPT-specific approximation** to
the author's hierarchy/neighbor routing, not an exact port. Only targets beyond
the actual advected outer velocity edges get zero inflow. Those outer edges
must remain graphs in x; folds fail. Decrease dt/remesh interval if geometry
or extension guards fail rather than loosening the guard without validation.

The AMR material C2 is constant between remeshes. At remesh, `remap_*` budgets
include **both** interpolation and the quadrature change from regridding.
`regrid_mass_change` and `regrid_c2_change` isolate the latter using the same
interpolant evaluated on the old partition; subtract them from `remap_*` for
the interpolation contribution. No post-remesh normalization is applied.

This baseline preallocates all candidate panels, at most 32768. Candidate
count is `roots * (1 + 4 + ... + 4**max_level)` and all candidates are sampled
at each remesh; remesh cost does not scale only with active leaves. Direct
push work still scales with **slot capacity**, including padding. Adaptivity
does not by itself establish a speedup. Initial nine-node indicators cannot
detect already unresolved beams or filaments: vary the base mesh, minimum
level, refinement tolerances and maximum level against refined references.
`remesh_every=0` selects an initial adaptive partition but never updates it.

`configs/farsight-1d/two-stream-amr.yaml` is a small adaptive execution example,
not a converged turbulence or general AMR-accuracy benchmark. It also has a
short independent linear-field regression, described below.

## Conservation and resolution

Diagnostics include mass, momentum, kinetic energy, electric energy, total
energy, $C_2=\sum_j W_j f_j^2$, minimum f, and integrated negative f. Signed and
absolute cumulative remesh changes in mass and C2 make interpolation errors
visible. Neither clipping nor mass/C2 renormalization hides those errors.

RK4 is not symplectic, and biquadratic remeshing is neither exactly conservative
nor positivity preserving. Between remeshes the constant material C2 is a
bookkeeping identity; it does not prove that deformed panels preserve volume
or resolve phase-space filaments. `max_panel_area_error` measures the maximum
relative change of straight-corner panel area **at remeshes only**, not the
Jacobian of the full curved flow map. With remeshing disabled, its zero value
means no such check was performed.

`electric_energy` uses the physical diagnostic $\frac12\int E^2 dx$ on the fixed
x nodes. Its sum with kinetic energy is not the exact interaction Hamiltonian
of the softened particle system. Study quadrature, softening, velocity cutoff,
timestep, and remesh resolution separately before making conservation or
turbulence claims.

For N phase-space nodes and P panels, direct field evaluation costs O(N²), and
panel search costs O(NP). Target batching bounds temporary pair arrays but does
not reduce this work. Reverse-mode differentiation can retain intermediates
across steps and requires substantially more memory. Large turbulence meshes
need an accelerated field solver and search before they are practical.

Softening must also be resolved by the source quadrature. Even a spatially
homogeneous distribution can acquire spurious off-node forces when moving
velocity layers sample an underresolved kernel. For the 32 by 64 two-stream
test mesh, with $L=2\pi/0.3$ and a free-streaming shift of $0.025v$, the maximum
field for homogeneous density is about 0.112 at epsilon 0.1, versus
$5.6\times10^{-7}$ at epsilon 1.5. The continuum field is zero in both cases.
This is tested explicitly. Reducing epsilon alone is not a convergence study;
refine the spatial quadrature with it.

## Run an example

The module CLI uses the explicit host runtime and writes NetCDF observations,
the complete final state as NPZ, a resolved manifest, and scalar metrics:

```bash
uv run python -m adept.farsight1d \
  --cfg configs/farsight-1d/two-stream.yaml \
  --output outputs/farsight-two-stream
```

The output directory must be new. The two included examples exercise linear
two-stream growth and Landau damping with deliberately resolved softening; they
are not turbulence configurations or epsilon-to-zero convergence studies. Add `--tracking-uri`
and optionally `--experiment` and `--name` to record metrics and upload artifacts
through the explicit MLflow services. Without that option outputs are local
and the run uses `NullTracker`. This new solver is not registered in the legacy
`ergoExo` / `run.py` dispatch.

For Python callers:

```python
import jax
import yaml
from adept import SimulationSpec, run_prepared, solver_registry

jax.config.update("jax_enable_x64", True)
with open("configs/farsight-1d/two-stream.yaml") as stream:
    config = yaml.safe_load(stream)
prepared = solver_registry.prepare(SimulationSpec.from_legacy_config(config), key=42)
completed = run_prepared(prepared, key=jax.random.key(42))
scalars = completed.report.result["scalars"]
fields = completed.report.result["fields"]
```

The builder itself has no logging or file-writing side effects. Its analyzer
returns datasets and metrics; the CLI's host analyzer adapter handles file
serialization. `LocalExecutor` / `RunPlan` can also dispatch the registered solver.
`run.json` is a local run summary written after completion; the six numerical
files written by the analyzer are the artifacts uploaded when tracking is enabled.

The numerical boundary accepts explicit `program`, `params`, `state`, `inputs`,
and `key` PyTrees. The current builder provides empty params and inputs; there
are no drivers or runtime parameter controls. Differentiate a chosen initial
distribution perturbation by constructing the state inside a JAX objective
and passing it into the program. Hard panel selection is piecewise
differentiable; gradients at ownership changes, boundary crossings or invalid
panels are not promised. Adaptive topology is selected by hard decisions;
derivatives describe only the current selection, not refinement sensitivity.
Batching, sharding and production
gradient validation are not advertised.

## Verification

`tests/test_farsight1d` checks the field against an independent scalar-loop sum,
periodic seams, quadrature, analytic remeshing, zero inflow, invalid panels,
remesh error accounting, JIT/grad, explicit host execution, and CLI artifact
readback. Adaptive checks include zero-level fixed-grid parity, mixed-level
quadrature, balancing, constant preservation through an interior interface gap,
coarsening, explicit overflow, and mass/C2 error-budget telescoping.
The characteristic integrator converges at fourth order against an
independent SciPy DOP853 solve; a smooth free-streaming remesh converges close
to third order in space. These operator checks do not establish fourth-order
accuracy of repeated remeshing in time.

Two short self-consistent regressions compare the **complete complex Fourier
field history**, including initial transients, against an independent linearized
Vlasov initial-value calculation. That reference uses a 513-node velocity
quadrature and the continuum Fourier multiplier of the same softened kernel.
At nx=32, nv=64, dt=0.05, amplitude=1e-4 and tmax=12, local float64 results were:

| Case | Relative L2 field-history error | Relative mass change | Relative C2 change |
| --- | --- | --- | --- |
| Two-stream (epsilon=1.5) | 0.689% | -1.71e-6 | -1.41e-4 |
| Maxwellian (epsilon=0.9) | 1.417% | -1.79e-6 | -2.37e-6 |

The parameterized test sets explicit acceptance bounds: field-history error
below 2%, absolute relative mass drift below 1e-4, and C2 drift below 1e-3.
The example YAMLs contain the matching physical and numerical parameters.
These comparisons validate these regularized linear problems; they do not
measure late-time turbulence, the unregularized limit, or production scaling.

The AMR two-stream example uses base 8 by 16 intervals, maximum level 2,
atol 0.05 and 512 slots, with the same physical parameters and time interval.
It retains 320 leaves (versus 512 at uniform finest refinement). Its complex
field-history error against the same linear reference is 0.836%, relative
mass change -5.89e-6 and C2 change -1.42e-4. The leaf set remains unchanged;
192 leaves report maximum-level saturation throughout. Final minimum f is
-1.70e-5 and integrated negative mass is 1.07e-4. This passes the same field
and invariant thresholds, but neither demonstrates converged refinement nor
validates a late-time turbulence case with substantial topology changes.

## Attribution

The numerical equations and method are credited to the paper above and the
[author implementation](https://github.com/RTSandberg/FARSIGHT). This JAX code
was written independently; it is not a source translation or an official port.
The implementation covers a bounded subset of the method, not the complete
feature set or performance of FARSIGHT.

See the [configuration reference](config.md) for every option and diagnostic.
