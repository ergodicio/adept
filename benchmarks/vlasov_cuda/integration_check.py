"""Exercise the native ADEPT/ergoExo path, original callbacks and NetCDF writers."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--vp-source", type=Path)
    parser.add_argument("--module", choices=["base", "vp", "both"], default="both")
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    import numpy as np
    import yaml
    from diffrax import SubSaveAt

    from adept import ergoExo
    from adept import patched_mlflow as mlflow
    from adept._vlasov1d.storage import store_f, store_fields
    from adept.vlasov1d import BaseVlasov1D

    args.output.mkdir(parents=True, exist_ok=False)
    mlflow.set_tracking_uri((args.output / "mlruns").as_uri())
    classes = [] if args.module == "vp" else [BaseVlasov1D]
    if args.vp_source:
        sys.path.insert(0, str(args.vp_source))
        from vp_turbulence.adeptmodules import VPTwoStream1D

        if args.module != "base":
            classes.append(VPTwoStream1D)
    checks = []
    for cls in classes:
        for ranks in args.ranks:
            cfg = yaml.safe_load(Path("tests/test_base/configs/example.yaml").read_text())
            cfg["grid"].update(nx=32, nv=64, dt=0.02, tmax=0.08, vmax=8.0)
            cfg["drivers"] = {"ex": {}, "ey": {}}
            cfg["terms"]["time"] = "sixth"
            cfg["terms"]["fokker_planck"].update(is_on=True, type="chang_cooper_dougherty")
            cfg["terms"]["fokker_planck"]["time"]["baseline"] = 0.001
            cfg["terms"]["fokker_planck"]["space"]["baseline"] = 1.0
            cfg["terms"]["krook"]["is_on"] = False
            cfg["save"] = {"fields": {"t": {"nt": 5}}, "electron": {"main": {"t": {"nt": 3}}}}
            beam = cfg["density"]["species-background"]
            beam.update(v0=3.0, T0=0.2, noise_val=0.001)
            cfg["density"]["species-left"] = copy.deepcopy(beam)
            cfg["density"]["species-left"].update(v0=-3.0, noise_seed=43)
            cfg["native"] = dict(workdir=str(args.output))
            if cls.__name__ == "VPTwoStream1D":
                cfg["two_stream"] = dict(
                    v_b=3.0,
                    delta_v_b=0.5,
                    native_cascade_budget=True,
                    native_kv_spectrum=dict(is_on=True, t=dict(tmin=0.0, tmax=0.1, nt=3)),
                )
                cfg["postprocess"] = {
                    key: False
                    for key in [
                        "plot_scalars",
                        "plot_fields",
                        "plot_fullres_dists",
                        "plot_lowres_dists",
                        "plot_transformed_dists",
                        "plot_field_spectrum",
                    ]
                }
            original = cls(copy.deepcopy(cfg))
            original.write_units()
            original.get_derived_quantities()
            original.get_solver_quantities()
            original.init_state_and_args()
            original.init_diffeqsolve()
            nodes = 2 if ranks > 4 else 1
            cfg["backend"] = "native"
            cfg["native"].update(
                device="cuda",
                library=str(args.library),
                ranks=ranks,
                timeout=180,
                launcher=[
                    "srun",
                    f"--jobid={os.environ['SLURM_JOB_ID']}",
                    "--overlap",
                    f"--nodes={nodes}",
                    "--ntasks={ranks}",
                    f"--ntasks-per-node={ranks // nodes}",
                    "--gpus-per-node=4",
                    "--gpu-bind=none",
                    "--cpu-bind=none",
                ],
            )
            cfg["mlflow"] = dict(experiment="native-adept-integration", run=f"{cls.__name__}-{ranks}")
            exo = ergoExo()
            modules = exo.setup(copy.deepcopy(cfg), adept_module=cls)
            output, post, run_id = exo(modules)
            for key, save in exo.adept_module.diffeqsolve_quants["saveat"]["subs"].items():
                subs = original.diffeqsolve_quants["saveat"]["subs"]
                subs[key] = SubSaveAt(ts=save.ts, fn=subs[key].fn)
            reference = original({})["solver result"]
            root = args.output / f"{cls.__name__}-{ranks}"
            (root / "binary").mkdir(parents=True)
            sol = output["solver result"]
            store_fields(exo.adept_module.cfg, str(root / "binary"), sol.ys["fields"], sol.ts["fields"], "fields")
            store_f(exo.adept_module.cfg, sol.ts, str(root), sol.ys)
            errors = []
            import jax

            for key in sol.ys:
                np.testing.assert_allclose(reference.ts[key], sol.ts[key], atol=1e-14)
                for a, b in zip(jax.tree.leaves(reference.ys[key]), jax.tree.leaves(sol.ys[key]), strict=True):
                    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-11, equal_nan=True, err_msg=key)
                    delta = np.not_equal(a, b) if np.asarray(a).dtype.kind == "b" else np.asarray(a) - np.asarray(b)
                    errors.append(float(np.max(np.abs(delta))) if delta.size else 0.0)
            row = dict(
                module=cls.__name__,
                ranks=ranks,
                passed=True,
                max_abs_error=max(errors),
                mlflow_run_id=run_id,
                stats=output["native stats"],
                keys=list(sol.ys),
            )
            checks.append(row)
            (args.output / "checks.json").write_text(json.dumps(checks, indent=2))
            print(json.dumps(row), flush=True)
    (args.output / "checks.json").write_text(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
