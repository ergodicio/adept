import subprocess
import sys
import textwrap


def run_isolated_python(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_importing_adept_contracts_does_not_load_numerical_or_tracking_dependencies():
    process = run_isolated_python(
        """
        import sys

        import adept
        from adept.core import (
            Executor,
            LocalExecutor,
            Artifact,
            DirectoryArtifactSink,
            MaterializedResult,
            Objective,
            ObjectiveResult,
            ObservationPlan,
            ObservationSchedule,
            Report,
            RunRequest,
            RunPlan,
            SimulationSpec,
            SolverRegistry,
            run_prepared,
        )

        forbidden = ("jax", "equinox", "mlflow", "diffrax")
        loaded = sorted(
            name for name in sys.modules
            if any(name == dependency or name.startswith(dependency + ".") for dependency in forbidden)
        )
        assert loaded == [], loaded
        assert adept.SimulationSpec is SimulationSpec
        assert adept.SolverRegistry is SolverRegistry
        assert adept.Objective is Objective
        assert adept.ObjectiveResult is ObjectiveResult
        assert adept.ObservationPlan is ObservationPlan
        assert adept.ObservationSchedule is ObservationSchedule
        assert adept.MaterializedResult is MaterializedResult
        assert adept.Artifact is Artifact
        assert adept.Executor is Executor
        assert adept.LocalExecutor is LocalExecutor
        assert adept.Report is Report
        assert adept.RunRequest is RunRequest
        assert adept.RunPlan is RunPlan
        assert adept.DirectoryArtifactSink is DirectoryArtifactSink
        assert adept.run_prepared is run_prepared
        assert "adept.core.objectives" not in sys.modules
        assert "adept.core.materialization" not in sys.modules
        assert adept.solver_registry.names() == ("pic-1d", "tf-1d")
        """
    )

    assert process.returncode == 0, process.stderr


def test_local_executor_bootstraps_x64_before_loading_a_solver_builder():
    process = run_isolated_python(
        """
        import os
        import sys

        os.environ.pop("JAX_ENABLE_X64", None)

        from adept import (
            ExecutionKind,
            LocalExecutor,
            Placement,
            Precision,
            PreparedSimulation,
            RawResult,
            Report,
            RunManifest,
            RunPlan,
            SimulationSpec,
            SolverCapabilities,
            SolverRegistry,
        )

        assert "jax" not in sys.modules
        capabilities = SolverCapabilities(
            ExecutionKind.DISCRETE,
            precision=Precision.X64,
            placements={Placement.SINGLE_DEVICE},
        )


        class Program:
            def __call__(self, params, state, inputs, key):
                del params, inputs, key
                return RawResult(state, {}, {}, "ok", {})


        class Analyzer:
            def analyze(self, result, manifest):
                del manifest
                return Report(result=result.final_state)


        class Builder:
            def prepare(self, spec, *, key):
                del spec, key
                assert "jax" in sys.modules
                import jax
                assert jax.config.read("jax_enable_x64")
                return PreparedSimulation(
                    program=Program(),
                    params={},
                    state=1,
                    inputs={},
                    manifest=RunManifest(
                        raw_config={},
                        resolved_config={},
                        structural_fingerprint="sha256:test",
                    ),
                    analyzer=Analyzer(),
                    capabilities=capabilities,
                )


        registry = SolverRegistry()
        registry.register_lazy("fake", Builder, capabilities=capabilities)

        def direct(prepared, key):
            return prepared.program(prepared.params, prepared.state, prepared.inputs, key)

        with LocalExecutor(registry=registry, execute_prepared=direct) as executor:
            plan = RunPlan(SimulationSpec("fake", {}))
            executor.validate(plan)
            assert "jax" not in sys.modules
            completed = executor.execute(plan)

        assert completed.report.result == 1
        assert os.environ["JAX_ENABLE_X64"] == "true"
        """
    )

    assert process.returncode == 0, process.stderr


def test_resolving_builtin_builders_does_not_load_mlflow():
    process = run_isolated_python(
        """
        import sys

        from adept.core import solver_registry

        assert solver_registry.resolve("tf-1d").__class__.__name__ == "TwoFluid1DBuilder"
        assert "mlflow" not in sys.modules
        assert solver_registry.resolve("pic-1d").__class__.__name__ == "PIC1DBuilder"
        assert "mlflow" not in sys.modules
        """
    )

    assert process.returncode == 0, process.stderr


def test_loading_objective_helpers_does_not_load_mlflow():
    process = run_isolated_python(
        """
        import sys

        from adept import CallableObjective, WeightedSumObjective, value_and_grad

        assert CallableObjective.__name__ == "CallableObjective"
        assert WeightedSumObjective.__name__ == "WeightedSumObjective"
        assert callable(value_and_grad)
        assert "mlflow" not in sys.modules
        """
    )

    assert process.returncode == 0, process.stderr


def test_loading_mlflow_adapter_classes_does_not_import_mlflow_until_used():
    process = run_isolated_python(
        """
        import sys

        from adept import MLflowArtifactSink, MLflowTracker

        assert MLflowTracker.__name__ == "MLflowTracker"
        assert MLflowArtifactSink.__name__ == "MLflowArtifactSink"
        assert "mlflow" not in sys.modules
        """
    )

    assert process.returncode == 0, process.stderr


def test_untracked_host_runtime_operates_when_mlflow_import_is_blocked():
    process = run_isolated_python(
        """
        import sys
        import tempfile


        class BlockMLflow:
            def find_spec(self, fullname, path, target=None):
                del path, target
                if fullname == "mlflow" or fullname.startswith("mlflow."):
                    raise AssertionError(f"unexpected MLflow import: {fullname}")
                return None


        sys.meta_path.insert(0, BlockMLflow())

        from adept import (
            DirectoryArtifactSink,
            ExecutionKind,
            PreparedSimulation,
            RawResult,
            RunManifest,
            SolverCapabilities,
            run_prepared,
        )


        class Program:
            def __call__(self, params, state, inputs, key):
                del params, state, inputs, key
                return RawResult(1, (), (), "ok", {})


        class Analyzer:
            def analyze(self, result, manifest):
                del manifest
                from adept import Report
                return Report(result=result)


        prepared = PreparedSimulation(
            program=Program(),
            params={},
            state={},
            inputs={},
            manifest=RunManifest(
                raw_config={},
                resolved_config={},
                structural_fingerprint="sha256:test",
            ),
            analyzer=Analyzer(),
            capabilities=SolverCapabilities(ExecutionKind.DISCRETE),
        )
        with tempfile.TemporaryDirectory() as directory:
            completed = run_prepared(
                prepared,
                key="key",
                artifact_sink=DirectoryArtifactSink(directory),
                execute=lambda simulation, key: simulation.program(
                    simulation.params,
                    simulation.state,
                    simulation.inputs,
                    key,
                ),
            )
        assert completed.raw_result.final_state == 1
        assert "mlflow" not in sys.modules
        """
    )

    assert process.returncode == 0, process.stderr


def test_legacy_public_symbols_still_resolve_lazily():
    process = run_isolated_python(
        """
        import sys

        import adept

        assert "adept._base_" not in sys.modules
        from adept import ADEPTModule, ergoExo
        assert "adept._base_" in sys.modules
        assert ADEPTModule.__name__ == "ADEPTModule"
        assert ergoExo.__name__ == "ergoExo"
        """
    )

    assert process.returncode == 0, process.stderr


def test_existing_eager_solver_attributes_remain_available():
    process = run_isolated_python(
        """
        import adept

        assert adept.vlasov1d.__name__ == "adept.vlasov1d"
        assert adept.vfp2d.__name__ == "adept.vfp2d"
        """
    )

    assert process.returncode == 0, process.stderr
