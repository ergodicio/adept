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
        from adept.core import Objective, ObjectiveResult, SimulationSpec, SolverRegistry

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
        assert "adept.core.objectives" not in sys.modules
        assert adept.solver_registry.names() == ("pic-1d", "tf-1d")
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
