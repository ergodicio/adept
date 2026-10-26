"""Native execution of existing ADEPT modules with their original save functions."""

import gc
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import jax
import numpy as np
from diffrax import RESULTS, Solution, SubSaveAt

from adept._vlasov1d.datamodel import NativeSolverConfig
from adept._vlasov1d.modules import BaseVlasov1D


@contextmanager
def _host_preprocessing():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _restore_array(host):
    """Recreate callback constants as replicated JAX arrays on the worker mesh."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    sharding = NamedSharding(Mesh(np.array(jax.devices()), ("device",)), PartitionSpec())
    return jax.make_array_from_callback(host.shape, sharding, lambda index: host[index])


def _host_pickle(payload, stream):
    import cloudpickle

    class HostPickler(cloudpickle.CloudPickler):
        def reducer_override(self, value):
            if isinstance(value, jax.Array):
                return _restore_array, (np.asarray(value),)
            return super().reducer_override(value)

    HostPickler(stream).dump(payload)


def _write_vx(path, values):
    """Export a host x-v state with bounded transpose temporaries."""
    output = np.memmap(path, mode="w+", dtype=np.float64, shape=values.shape[::-1])
    for start in range(0, values.shape[1], 64):
        output[start : start + 64] = np.asarray(values[:, start : start + 64]).T
    output.flush()
    del output


def _collect(root, times, ranks):
    import cloudpickle

    with (root / "layout.pkl").open("rb") as stream:
        layout = cloudpickle.load(stream)
    indices = [json.loads((root / f"rank-{rank:05d}/index.json").read_text()) for rank in range(ranks)]
    result = {}
    directory = root / "collected"
    directory.mkdir()
    for key_id, (key, definition) in enumerate(layout.items()):
        arrays = []
        for leaf_id, leaf in enumerate(definition["leaves"]):
            shape = (len(times[key]), *leaf["shape"])
            parts = [index[key][leaf_id] for index in indices if index[key][leaf_id] is not None]
            if len(parts) == 1 and all(
                start == 0 and stop == size and step == 1
                for (start, stop, step), size in zip(parts[0]["index"], leaf["shape"], strict=True)
            ):
                array = np.load(root / parts[0]["file"], mmap_mode="r")
            else:
                array = np.lib.format.open_memmap(
                    directory / f"{key_id:04d}-{leaf_id:03d}.npy", mode="w+", dtype=np.dtype(leaf["dtype"]), shape=shape
                )
                for part in parts:
                    local = np.load(root / part["file"], mmap_mode="r")
                    slices = tuple(slice(*axis) for axis in part["index"])
                    # Bound host copies for large saved distributions.
                    for frame in range(len(times[key])):
                        array[(frame, *slices)] = local[frame]
                array.flush()
            if array.shape != shape:
                raise RuntimeError(f"native save shape mismatch for {key}")
            arrays.append(array)
        result[key] = jax.tree.unflatten(definition["tree"], arrays)
    return result


class NativeVlasovMixin:
    """Swap execution while retaining a Vlasov module's lifecycle and callbacks."""

    jit_compile = False

    def __init__(self, cfg):
        if not jax.config.jax_enable_x64:
            raise RuntimeError("native Vlasov requires jax_enable_x64 before preprocessing")
        with _host_preprocessing():
            super().__init__(cfg)
        self.native_options = NativeSolverConfig.model_validate(cfg.get("native") or {})
        self._validate_native_physics()

    def _validate_native_physics(self):
        cfg, sim = self.cfg, self.simulation
        names = [species.name for species in sim.species]
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("native species names must be nonempty and unique")
        for species in sim.species:
            if not np.isfinite(species.mass) or species.mass <= 0 or not np.isfinite(species.charge):
                raise ValueError("native species require positive finite mass and finite charge")
            if not np.isclose(species.vmin, -species.vmax, rtol=0, atol=1e-12):
                raise NotImplementedError("native Vlasov requires symmetric velocity domains")
        terms = cfg["terms"]
        if terms["time"] != "sixth" or terms["edfdv"] != "cubic-spline" or terms["field"] != "poisson":
            raise NotImplementedError(
                "native Vlasov requires sixth-order time, cubic-spline velocity, and Poisson fields"
            )
        if terms.get("vdfdx", "exponential") != "exponential":
            raise NotImplementedError("native Vlasov requires exponential spatial streaming")
        if (
            any(cfg["drivers"][key] for key in ["ex", "ey"])
            or cfg["drivers"].get("ex_stochastic") is not None
            or terms["krook"]["is_on"]
            or terms.get("hou_li_filter", {}).get("is_on", False)
        ):
            raise NotImplementedError("native Vlasov does not support external drivers, Krook relaxation, or filtering")
        if any(cfg.get("diagnostics", {}).values()):
            raise NotImplementedError("native Vlasov does not yet evolve df/dt diagnostic state")
        fp = terms["fokker_planck"]
        self.native_collision_frequency = 0.0
        if fp["is_on"]:
            if fp["type"].casefold() not in {"chang_cooper_dougherty", "dougherty_chang_cooper"} or (
                fp.get("self_consistent_beta") or {}
            ).get("enabled", False):
                raise NotImplementedError("native collisions require Chang-Cooper Dougherty with analytic beta")
            for axis in ["time", "space"]:
                envelope = getattr(sim.nu_fp_prof, f"{axis}_envelope")
                resolved_config = getattr(self.config_model.terms.fokker_planck, axis)
                if envelope.bump_height != 0 or float(resolved_config.slope) != 0:
                    raise NotImplementedError("native collisions currently require constant frequency profiles")
            self.native_collision_frequency = float(
                sim.nu_fp_prof.time_envelope.baseline * sim.nu_fp_prof.space_envelope.baseline
            )

    def write_units(self):
        with _host_preprocessing():
            return super().write_units()

    def get_derived_quantities(self):
        with _host_preprocessing():
            return super().get_derived_quantities()

    def get_solver_quantities(self):
        with _host_preprocessing():
            return super().get_solver_quantities()

    def init_state_and_args(self):
        with _host_preprocessing():
            return super().init_state_and_args()

    def init_diffeqsolve(self):
        with _host_preprocessing():
            super().init_diffeqsolve()
        # Standard outputs use native steps, avoiding full-state copies for
        # Diffrax's off-step interpolation. Keep sparse subclass callback times
        # unless the caller opts into step rounding for those too; some existing
        # project postprocessors reconstruct those axes from their own config.
        t0, t1 = (float(self.time_quantities[key]) for key in ["t0", "t1"])
        dt = self.simulation.grid.dt
        steps = round((t1 - t0) / dt)
        subs = self.diffeqsolve_quants["saveat"]["subs"]
        for key, save in list(subs.items()):
            standard = key in self.cfg["save"]
            if key == "default":
                indices = np.arange(0, steps + 1, self.native_options.scalar_stride, dtype=int)
                indices = np.union1d(indices, [steps])
            elif standard or self.native_options.custom_save_times == "steps":
                requested = np.asarray(save.ts)
                if (
                    requested.size == 0
                    or np.any(np.diff(requested) < 0)
                    or requested.min() < t0 - 1e-9
                    or requested.max() > t1 + 1e-9
                ):
                    raise ValueError("native save times must be ordered and inside the solve interval")
                indices = np.unique(np.clip(np.rint((requested - t0) / dt).astype(int), 0, steps))
            else:
                continue
            times = t0 + indices * dt
            subs[key] = SubSaveAt(ts=times, fn=save.fn)
            if standard:
                self.cfg["save"][key]["t"].update(ax=times, nt=len(times), tmin=float(times[0]), tmax=float(times[-1]))

    def __call__(self, trainable_modules=None, args=None):
        if jax.process_count() != 1:
            raise RuntimeError("run one outer Python driver; the native backend launches its MPI workers")
        if trainable_modules:
            raise NotImplementedError("native Vlasov does not execute trainable modules")
        if args is not None and args is not self.args:
            raise NotImplementedError("native Vlasov currently uses the configured static arguments")
        options = self.native_options
        library = options.library or os.environ.get("ADEPT_VLASOV_LIBRARY")
        if not library or not Path(library).expanduser().is_file():
            raise FileNotFoundError(
                "Build adept/_vlasov1d/native with CMake and set native.library or ADEPT_VLASOV_LIBRARY"
            )
        library = str(Path(library).expanduser().resolve())
        grid = self.simulation.grid
        t0, t1 = (float(self.time_quantities[key]) for key in ["t0", "t1"])
        steps = round((t1 - t0) / grid.dt)
        if steps < 1 or not np.isclose(steps * grid.dt, t1 - t0, rtol=0, atol=1e-9):
            raise NotImplementedError("native solve endpoints must be separated by an integer number of timesteps")
        species_list = sorted(self.simulation.species, key=lambda species: species.name != "electron")
        if grid.nx % options.ranks or any(species.nv % options.ranks for species in species_list):
            raise ValueError("native ranks must divide nx and every species nv")
        species_names = {species.name for species in species_list}
        known = species_names | {"e", "de", "a", "da", "prev_a"}
        if set(self.state) - known or any(np.any(np.asarray(self.state[key])) for key in ["de", "a", "da", "prev_a"]):
            raise NotImplementedError(
                "native Vlasov requires electrostatic state without additional evolved diagnostics"
            )
        subs = self.diffeqsolve_quants["saveat"]["subs"]
        times = {key: np.asarray(save.ts, dtype=float).tolist() for key, save in subs.items()}
        if any(
            not values or min(values) < t0 - 1e-9 or max(values) > t1 + 1e-9 or np.any(np.diff(values) < 0)
            for values in times.values()
        ):
            raise ValueError("native save times must be ordered and inside the solve interval")
        base = options.workdir or os.environ.get("BASE_TEMPDIR")
        if options.ranks > 1 and not base:
            raise ValueError("set native.workdir to a shared directory for MPI workers")
        if base:
            base = os.path.expandvars(os.path.expanduser(base))
            if "$" in base:
                raise ValueError("native.workdir contains an undefined environment variable")
            Path(base).mkdir(parents=True, exist_ok=True)
        workspace = tempfile.TemporaryDirectory(prefix="adept-native-", dir=base)
        root = Path(workspace.name)
        # Retain backing files for the NumPy memmaps returned in solver result.
        self._native_workspaces = getattr(self, "_native_workspaces", []) + [workspace]
        for name, values in [
            ("background.bin", self.cfg["grid"]["ion_charge"]),
            ("field.bin", self.state["e"]),
        ]:
            np.asarray(values, dtype=np.float64).tofile(root / name)
        # Initializer cache is not used by save callbacks. Avoid serializing a
        # second full distribution through a closure over the grid config.
        cache = self.cfg["grid"].pop("species_distributions", None)
        try:
            with (root / "callbacks.pkl").open("wb") as stream:
                _host_pickle(
                    dict(
                        callbacks={key: save.fn for key, save in subs.items()},
                        args=self.args,
                        state={key: value for key, value in self.state.items() if key not in species_names | {"e"}},
                    ),
                    stream,
                )
        finally:
            if cache is not None:
                self.cfg["grid"]["species_distributions"] = cache
        nu = self.native_collision_frequency
        common_args = [
            "--nx",
            str(grid.nx),
            "--length",
            str(grid.xmax - grid.xmin),
            "--dt",
            str(grid.dt),
            "--cells",
            "--integrator",
            "sixth",
            "--velocity",
            "cubic",
            "--layout",
            "halos",
            "--halo-capacity",
            str(options.halo_capacity),
            "--background",
            str(root / "background.bin"),
        ]
        if options.host_mpi:
            common_args.append("--host-mpi")
        species_specs = []
        for index, species in enumerate(species_list):
            species_grid = self.cfg["grid"]["species_grids"][species.name]
            initial, coordinates = root / f"initial-{index}.bin", root / f"v-{index}.bin"
            _write_vx(initial, self.state[species.name])
            np.asarray(species_grid["v"], dtype=np.float64).tofile(coordinates)
            species_args = [
                *common_args,
                "--nv",
                str(species.nv),
                "--vmax",
                str(species.vmax),
                "--charge",
                str(species.charge),
                "--mass",
                str(species.mass),
                "--nu",
                str(nu if index == 0 else 0.0),
                "--input",
                str(initial),
                "--v-grid",
                str(coordinates),
            ]
            species_specs.append(
                dict(
                    name=species.name,
                    nv=species.nv,
                    vmax=species.vmax,
                    charge=species.charge,
                    mass=species.mass,
                    options=species_args,
                )
            )
        request = dict(
            library=library,
            options=species_specs[0]["options"],
            species=species_specs,
            times=times,
            t0=t0,
            dt=grid.dt,
            steps=steps,
            device=options.device,
            ranks=options.ranks,
            pythonpath=[str(Path(entry or os.getcwd()).resolve()) for entry in sys.path],
        )
        (root / "request.json").write_text(json.dumps(request, indent=2))
        launcher = [part.format(ranks=options.ranks) for part in options.launcher]
        if options.ranks > 1 and not launcher:
            launcher = ["mpiexec", "-n", str(options.ranks)]
        command = [*launcher, sys.executable, str(Path(__file__).with_name("worker.py")), str(root / "request.json")]
        environment = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
        environment.setdefault("MPICH_GPU_SUPPORT_ENABLED", "1")
        log = root / "worker.log"
        with log.open("w") as stream:
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT, env=environment, start_new_session=True
            )
            try:
                returncode = process.wait(timeout=options.timeout)
            except BaseException as error:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(
                    f"native Vlasov workers interrupted; log: {log}\n{log.read_text()[-12000:]}"
                ) from error
        if returncode:
            raise RuntimeError(f"native Vlasov workers failed ({returncode}); log: {log}\n{log.read_text()[-12000:]}")
        ys = _collect(root, times, options.ranks)
        stats = json.loads((root / "stats.json").read_text())
        result = Solution(
            t0=t0,
            t1=t1,
            ts={key: np.asarray(value) for key, value in times.items()},
            ys=ys,
            stats={"num_steps": steps, "num_accepted_steps": steps, "num_rejected_steps": 0},
            result=RESULTS.successful,
            interpolation=None,
            solver_state=None,
            controller_state=None,
            made_jump=None,
            event_mask=None,
        )
        gc.collect()
        return {"solver result": result, "native stats": stats, "native workdir": str(root)}

    def post_process(self, run_output, td):
        source = Path(run_output["native workdir"])
        directory = Path(td) / "native"
        directory.mkdir(exist_ok=True)
        for name in ["request.json", "stats.json", "worker.log"]:
            shutil.copyfile(source / name, directory / name)
        request = json.loads((source / "request.json").read_text())
        library = Path(request["library"])
        (directory / "library.sha256").write_text(
            f"{hashlib.sha256(library.read_bytes()).hexdigest()}  {library.name}\n"
        )
        with _host_preprocessing():
            return super().post_process(run_output, td)

    def vg(self, trainable_modules, args):
        raise NotImplementedError("the native Vlasov backend does not provide automatic differentiation")


@cache
def native_backend(module_class):
    """Adapt a BaseVlasov1D subclass without changing its pre/postprocessing."""
    if not isinstance(module_class, type) or not issubclass(module_class, BaseVlasov1D):
        raise TypeError("the native backend requires a BaseVlasov1D subclass")
    if issubclass(module_class, NativeVlasovMixin):
        return module_class
    return type(f"Native{module_class.__name__}", (NativeVlasovMixin, module_class), {})


NativeVlasov1D = native_backend(BaseVlasov1D)
