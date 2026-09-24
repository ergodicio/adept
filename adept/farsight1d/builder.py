"""Logging-free preparation and host analysis for the FARSIGHT-inspired solver."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from adept.core import (
    MetricEvent,
    ObservationPlan,
    ObservationSchedule,
    PreparedSimulation,
    Report,
    RunManifest,
    SimulationSpec,
)
from adept.core.builtin_solvers import FARSIGHT1D_CAPABILITIES
from adept.core.observations_jax import infer_observation_spec
from adept.core.preparation import normalize_key, structural_fingerprint
from adept.core.programs import ScanProgram
from adept.farsight1d.amr import AdaptiveFarsightSystem, initialize_amr, make_hierarchy
from adept.farsight1d.config import Farsight1DConfig, SaveCadence
from adept.farsight1d.numerics import FarsightSystem, diagnose, electric_field, initial_state, make_mesh
from adept.farsight1d.positivity import initialize_positivity
from adept.farsight1d.treecode import TreecodeField


class FarsightProgram(ScanProgram):
    """Expose invalid geometry, capacity exhaustion, or nonfinite evolution."""

    def __call__(self, params, state, inputs, key):
        if not isinstance(params, dict) or params or not isinstance(inputs, dict) or inputs:
            raise ValueError("farsight-1d currently requires empty params and inputs; differentiate the initial state")
        result = super().__call__(params, state, inputs, key)
        return result._replace(
            status=diffrax.RESULTS.where(
                result.final_state["valid"], diffrax.RESULTS.successful, diffrax.RESULTS.nonfinite
            )
        )


class FarsightFieldsObservation(eqx.Module):
    """Regularized electric field evaluated at fixed, unique periodic x nodes."""

    x: jax.Array
    weights: jax.Array
    length: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    field_solver: Any = None

    def __call__(self, t: Any, state: Any, inputs: Any) -> dict[str, jax.Array]:
        del t, inputs
        evaluate = electric_field if self.field_solver is None else self.field_solver
        field = evaluate(
            self.x,
            state["x"].reshape(-1),
            (-state.get("weights", self.weights) * state["f"]).reshape(-1),
            self.length,
            self.epsilon,
            self.chunk_size,
        )
        return {"electric_field": field}


class FarsightScalarsObservation(eqx.Module):
    """Quadrature invariants, remesh error budget, and physical field energy."""

    fields: FarsightFieldsObservation

    def __call__(self, t: Any, state: Any, inputs: Any) -> dict[str, jax.Array]:
        result = diagnose(state, state.get("weights", self.fields.weights))
        field = self.fields(t, state, inputs)["electric_field"]
        result["electric_energy"] = 0.5 * self.fields.length * jnp.mean(field**2)
        result["total_energy"] = result["kinetic_energy"] + result["electric_energy"]
        return result


class FarsightDistributionObservation(eqx.Module):
    """Moving phase-space node coordinates and their attached distribution values."""

    def __call__(self, t: Any, state: Any, inputs: Any) -> dict[str, jax.Array]:
        del t, inputs
        names = ("x", "v", "f", "weights", "active", "panel_id", "level") if "active" in state else ("x", "v", "f")
        return {name: state[name] for name in names}


@dataclass(frozen=True)
class Farsight1DAnalyzer:
    """Convert retained observations to labeled datasets without ambient file IO."""

    def analyze(self, result, manifest):
        import xarray as xr

        if not bool(np.asarray(result.final_state["valid"])):
            raise ArithmeticError(
                "FARSIGHT evolution encountered invalid panel geometry or nonfinite values, "
                "AMR capacity exhaustion, or a positivity-limiter failure; "
                "reduce dt/remesh_every and inspect invalid_panels/max_panel_area_error; "
                "for capacity_exceeded increase amr.max_panels and inspect requested_panels; "
                "for remap_gap_nodes inspect max_gap_fraction against amr.max_gap_fraction and refine the mesh; "
                "for positivity_failed_panels inspect nonnegative finite panel means and limiter-stage budgets, "
                "then refine the mesh or reduce dt/remesh_every; a negative panel mean cannot be made "
                "nonnegative while preserving its mass"
            )
        numerical_values = (result.final_state, result.observations, result.times, result.stats)
        if any(not np.all(np.isfinite(np.asarray(value))) for value in jax.tree.leaves(numerical_values)):
            raise ArithmeticError("FARSIGHT result contains nonfinite state, observations, or statistics")

        grid = manifest.resolved_config["grid"]
        datasets = {}
        for name, observations in result.observations.items():
            times = np.asarray(result.times[name])
            if name == "scalars":
                datasets[name] = xr.Dataset(
                    {key: ("t", np.asarray(value)) for key, value in observations.items()},
                    coords={"t": times},
                )
                for key in ("electric_energy", "total_energy"):
                    datasets[name][key].attrs["description"] = (
                        "Physical-grid E^2/2 diagnostic; not the exact regularized interaction Hamiltonian"
                    )
                datasets[name]["c2"].attrs["description"] = (
                    "Native nodal reference-quadrature sum of f^2, including both signs of f; "
                    "c2 = c2_positive + c2_negative. This is not an exact integral of the squared "
                    "panel polynomial, and constancy does not establish positivity or resolution"
                )
                descriptions = {
                    "c2_positive": "Native nodal reference-quadrature sum W*max(f,0)^2; not clipped total C2",
                    "c2_negative": (
                        "Native nodal reference-quadrature sum W*min(f,0)^2; a nonnegative contribution to C2, "
                        "not a signed subtraction or an integral over negative polynomial regions"
                    ),
                    "positive_mass": "Native nodal sum W*max(f,0); mass = positive_mass - negative_mass",
                    "negative_mass": ("Native nodal sum W*max(-f,0); does not detect undershoot between stored nodes"),
                    "negative_node_count": (
                        "Unweighted number of negative stored nodes, excluding inactive AMR slots but "
                        "including independent active panel-edge traces"
                    ),
                    "min_f": "Minimum active stored nodal value; not a lower bound of the panel polynomial",
                    "initial_positivity_limited_panels": "Panels limited once after initial mass normalization",
                    "initial_positivity_failed_panels": "Initial panels failing the positivity-limiter validity check",
                    "initial_positivity_min_theta": "Minimum initial Bernstein rescaling factor; one means unchanged",
                    "initial_positivity_polynomial_c2_change": (
                        "Exact rectangular-panel integral change of f^2 from the initial Bernstein limiter; "
                        "separate from the native nodal C2 change and excluded from evolution budgets"
                    ),
                    "source_limiter_panels": "Cumulative source panels rescaled at remesh candidate locations",
                    "destination_limiter_panels": "Cumulative destination panels limited after leaf selection",
                    "positivity_failed_panels": "Cumulative failed positivity-limiter panel checks during remeshing",
                    "source_limiter_min_theta": "Smallest source sample rescaling factor during evolution",
                    "destination_limiter_min_theta": "Smallest destination Bernstein rescaling factor during evolution",
                    "min_bernstein_coefficient": (
                        "Minimum Bernstein coefficient at the latest rectangular initialization/remesh; "
                        "not a positivity certificate for a subsequently advected physical-coordinate fit"
                    ),
                    "destination_limiter_polynomial_c2_change": (
                        "Cumulative exact rectangular-panel integral changes of f^2 from destination limiting; "
                        "not a term in the native nodal remap_c2_change decomposition"
                    ),
                }
                stages = {
                    "interpolation": "raw candidate interpolation evaluated on the previous leaf layout",
                    "source_limiter": "source sample limiting evaluated on the previous leaf layout",
                    "regrid": "selection of the new leaf partition from the sampled candidate values",
                    "destination_limiter": "Bernstein limiting on the selected rectangular destination panels",
                }
                for moment, expression in (("mass", "W*f"), ("c2", "W*f^2")):
                    descriptions[f"initial_positivity_{moment}_change"] = (
                        f"Native nodal sum {expression} after-minus-before initial limiting; "
                        "applied after normalization and excluded from cumulative evolution budgets"
                    )
                    for stage, description in stages.items():
                        descriptions[f"{stage}_{moment}_change"] = (
                            f"Cumulative native nodal sum {expression} changes from {description}; "
                            "interpolation + source_limiter + regrid + destination_limiter equals "
                            f"remap_{moment}_change when positivity limiting is enabled"
                        )
                for key, description in descriptions.items():
                    if key in datasets[name]:
                        datasets[name][key].attrs["description"] = description
            elif name == "fields":
                x_axis = np.linspace(grid["xmin"], grid["xmax"], grid["nx"], endpoint=False)
                datasets[name] = xr.Dataset(
                    {key: (("t", "x"), np.asarray(value)) for key, value in observations.items()},
                    coords={"t": times, "x": x_axis},
                )
            elif "active" in observations:
                panel_count, node_count = np.shape(observations["f"])[1:]
                datasets[name] = xr.Dataset(
                    {
                        key: (("t", "panel") if np.ndim(value) == 2 else ("t", "panel", "node"), np.asarray(value))
                        for key, value in observations.items()
                    },
                    coords={"t": times, "panel": np.arange(panel_count), "node": np.arange(node_count)},
                )
                datasets[name].attrs["description"] = (
                    "Packed adaptive biquadratic panels; active identifies occupied slots, panel_id identifies "
                    "the reference hierarchy cell, and weights are material quadrature weights. "
                    "x and v are moving node coordinates; inactive slots do not contribute. "
                    "A slot can identify a different panel after remeshing."
                )
            else:
                datasets[name] = xr.Dataset(
                    {key: (("t", "x_node", "v_node"), np.asarray(value)) for key, value in observations.items()},
                    coords={"t": times, "x_node": np.arange(grid["nx"] + 1), "v_node": np.arange(grid["nv"] + 1)},
                )
                datasets[name].attrs["description"] = (
                    "x and v are moving marker coordinates, not fixed coordinate axes; "
                    "the x endpoint duplicates the periodic seam with quadrature endpoint weights"
                )
            datasets[name].attrs["solver"] = "farsight-1d"
            datasets[name].attrs["normalization"] = "electron plasma units: n0=m_e=|q_e|=epsilon_0=1"
            limiter = manifest.resolved_config["numerical"].get("positivity_limiter", "none")
            datasets[name].attrs["positivity_limiter"] = limiter
            if limiter == "bernstein":
                datasets[name].attrs["positivity_scope"] = (
                    "Experimental: nonnegative remesh candidate samples and Bernstein-certified rectangular "
                    "destination polynomials, to roundoff; independent panel traces can disagree at edges. "
                    "Not a conservative remap, an advected-fit positivity guarantee, or a C2 repair."
                )

        scalars = result.observations["scalars"]
        values = {f"final_{name}": float(np.asarray(value)[-1]) for name, value in scalars.items()}
        for name in ("mass", "c2"):
            start, stop = np.asarray(scalars[name])[[0, -1]]
            values[f"relative_{name}_change"] = float((stop - start) / start)
        return Report(result=datasets, metrics=(MetricEvent(values, step=int(np.asarray(result.stats["num_steps"]))),))


def _schedule(cadence: SaveCadence, num_steps: int) -> ObservationSchedule:
    points = tuple(range(0, num_steps + 1, cadence.every_steps))
    if points[-1] != num_steps:
        points += (num_steps,)
    return ObservationSchedule.at_steps(points)


def _initial_distribution(config: Farsight1DConfig, x, v, weights):
    initial = config.initial
    velocity_shape = jnp.exp(-0.5 * ((v - initial.drift) / initial.thermal_speed) ** 2)
    if initial.kind == "two-stream":
        velocity_shape += jnp.exp(-0.5 * ((v + initial.drift) / initial.thermal_speed) ** 2)
    length = config.grid.xmax - config.grid.xmin
    spatial_shape = 1.0 + initial.amplitude * jnp.cos(2 * jnp.pi * initial.mode * (x - config.grid.xmin) / length)
    distribution = spatial_shape * velocity_shape
    mass = jnp.sum(distribution * weights)
    if not np.isfinite(float(mass)) or float(mass) <= 0:
        raise ValueError("Initial distribution is unresolved on the configured velocity domain and mesh")
    return (distribution / mass) * length


class Farsight1DBuilder:
    """Prepare a normalized electron solve using the explicit ADEPT host contracts."""

    def prepare(self, spec: SimulationSpec, *, key: int | jax.Array) -> PreparedSimulation:
        if spec.solver != "farsight-1d":
            raise ValueError(f"Farsight1DBuilder cannot prepare solver {spec.solver!r}")
        if spec.schema_version != "1":
            raise ValueError(f"Unsupported FARSIGHT specification schema version {spec.schema_version!r}")
        if not jax.config.jax_enable_x64:
            raise RuntimeError("farsight-1d requires float64; enable jax_enable_x64 before preparation")
        config = Farsight1DConfig.model_validate(spec.config_dict())
        normalized_key, seed, provenance = normalize_key(key)
        grid, time, numerical = config.grid, config.time, config.numerical
        length = grid.xmax - grid.xmin
        params, inputs = {}, {}
        field_solver = (
            TreecodeField(**numerical.treecode.model_dump()) if numerical.field_solver == "treecode" else None
        )
        system_options = {
            "length": length,
            "dt": time.dt,
            "epsilon": numerical.epsilon,
            "charge": -1.0,
            "mass": 1.0,
            "remesh_every": numerical.remesh_every,
            "chunk_size": numerical.chunk_size,
            "field_solver": field_solver,
        }
        if config.amr.enabled:
            amr = config.amr
            hierarchy = make_hierarchy(
                grid.nx, grid.nv, grid.xmin, grid.xmax, grid.vmin, grid.vmax, amr.max_level, numerical.quadrature
            )
            # Only one complete hierarchy level contributes to candidate
            # normalization; summing all levels would count the domain repeatedly.
            finest_weights = jnp.where((hierarchy.level == amr.max_level)[:, None], hierarchy.weights, 0.0)
            candidate_f = _initial_distribution(config, hierarchy.x, hierarchy.v, finest_weights)
            selection_options = {
                "max_panels": amr.max_panels,
                "min_level": amr.min_level,
                "atol": amr.atol,
                "rtol": amr.rtol,
            }
            state = initialize_amr(hierarchy, candidate_f, **selection_options)
            if bool(np.asarray(state["capacity_exceeded"])):
                requested = int(np.asarray(state["requested_panels"]))
                raise ValueError(
                    f"Initial AMR hierarchy requests {requested} leaf panels but amr.max_panels={amr.max_panels}; "
                    "increase amr.max_panels or relax amr.atol/amr.rtol"
                )
            # Normalize only the initial selected representation. Subsequent
            # remesh/regrid defects remain visible in the invariant budgets.
            selected_mass = jnp.sum(state["weights"] * state["f"])
            if not np.isfinite(float(selected_mass)) or float(selected_mass) <= 0:
                raise ValueError("Initial distribution is unresolved on the selected AMR panels")
            state = {**state, "f": state["f"] * (length / selected_mass)}
            if numerical.positivity_limiter == "bernstein":
                # The limiter preserves each selected panel's Simpson mass.
                # Never normalize after limiting: its defects must remain visible.
                state = initialize_positivity(state)
                if not bool(np.asarray(state["valid"])):
                    failed = int(np.asarray(state["initial_positivity_failed_panels"]))
                    raise ValueError(
                        f"Initial Bernstein positivity limiter failed on {failed} panels; "
                        "inspect finite, nonnegative panel means and refine the initial mesh. "
                        "A negative panel mean cannot be made nonnegative while preserving its mass"
                    )
            weights = state["weights"]
            system = AdaptiveFarsightSystem(
                hierarchy=hierarchy,
                max_gap_fraction=amr.max_gap_fraction,
                positivity_limiter=numerical.positivity_limiter,
                **selection_options,
                **system_options,
            )
        else:
            x, v, weights = make_mesh(
                grid.nx, grid.nv, grid.xmin, grid.xmax, grid.vmin, grid.vmax, numerical.quadrature
            )
            state = initial_state(x, v, _initial_distribution(config, x, v, weights), weights)
            system = FarsightSystem(x0=x, v0=v, weights=weights, **system_options)
        field_x = jnp.linspace(grid.xmin, grid.xmax, grid.nx + 1)[:-1]
        fields = FarsightFieldsObservation(
            field_x, weights, length, numerical.epsilon, numerical.chunk_size, field_solver
        )
        functions = {
            "scalars": FarsightScalarsObservation(fields),
            "fields": fields,
            "distribution": FarsightDistributionObservation(),
        }
        plan = ObservationPlan(
            tuple(
                infer_observation_spec(
                    name, functions[name], _schedule(cadence, time.num_steps), t=time.tmin, state=state, inputs=inputs
                )
                for name in functions
                if (cadence := getattr(config.save, name)) is not None
            )
        )
        program = FarsightProgram.from_observation_plan(
            system=system,
            plan=plan,
            state=state,
            inputs=inputs,
            t0=time.tmin,
            dt=time.dt,
            num_steps=time.num_steps,
        )
        resolved = config.model_dump()
        resolved["time"]["num_steps"] = time.num_steps
        units = {
            "normalization": "electron plasma units: n0=m_e=|q_e|=epsilon_0=1",
            "x": "v_ref / omega_pe",
            "v": "v_ref",
            "t": "1 / omega_pe",
            "electric_field": "m_e * v_ref * omega_pe / |q_e|",
            "background": "periodic kernel removes the instantaneous mean charge; no net periodic charge mode",
            "field_energy": "physical E^2/2 quadrature, not the exact regularized Hamiltonian",
        }
        if config.amr.enabled:
            units["amr_gradients"] = (
                "piecewise derivatives with refinement decisions fixed; thresholds, panel ownership, "
                "and capacity decisions are discrete"
            )
        if numerical.positivity_limiter == "bernstein":
            units["positivity_limiter"] = (
                "experimental candidate-sample and rectangular Bernstein limiting; destination Simpson mass "
                "is preserved without renormalization, but the full remap is not conservative; native nodal "
                "and polynomial C2 limiter defects are reported separately"
            )
        if field_solver is not None:
            units["treecode_gradients"] = (
                "theta=0 uses the direct field and supports reverse-mode derivatives"
                if numerical.treecode.theta == 0
                else "forward-mode derivatives only within fixed sorting, activity, and opening decisions; "
                "reverse-mode derivatives through the tree walk are unsupported"
            )
            units["treecode_conservation"] = (
                "with theta>0 the particle-cluster force approximation is not pair-symmetric; exact momentum/energy "
                "conservation is not enforced, and material C2 alone does not measure phase-space resolution"
            )
        manifest = RunManifest(
            raw_config=spec.config_dict(),
            resolved_config=resolved,
            units=units,
            seed=seed,
            key_provenance=provenance,
            structural_fingerprint=structural_fingerprint(spec, program, params, state, inputs, normalized_key),
        )
        return PreparedSimulation(
            program=program,
            params=params,
            state=state,
            inputs=inputs,
            manifest=manifest,
            analyzer=Farsight1DAnalyzer(),
            capabilities=replace(
                FARSIGHT1D_CAPABILITIES, differentiable=field_solver is None or numerical.treecode.theta == 0
            ),
            observation_plan=plan,
        )


__all__ = ["Farsight1DAnalyzer", "Farsight1DBuilder", "FarsightProgram"]
