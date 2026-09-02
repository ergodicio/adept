from dataclasses import dataclass

import pytest

from adept.core import (
    ExecutionKind,
    InvalidSolverNameError,
    PreparedSimulation,
    RunManifest,
    SimulationSpec,
    SolverAlreadyRegisteredError,
    SolverCapabilities,
    SolverRegistry,
    UnknownSolverError,
)


@dataclass
class FakeBuilder:
    calls: list[tuple[SimulationSpec, object]]

    def prepare(self, spec: SimulationSpec, *, key: object) -> PreparedSimulation:
        self.calls.append((spec, key))
        return PreparedSimulation(
            program="program",
            params={"scale": 1.0},
            state={"value": 0.0},
            inputs=spec.config_dict(),
            manifest=RunManifest(
                raw_config=spec.config,
                resolved_config=spec.config,
                structural_fingerprint="sha256:fake",
                key_provenance=str(key),
            ),
            analyzer="analyzer",
            capabilities=SolverCapabilities(ExecutionKind.CONTINUOUS, differentiable=True),
        )


def test_registry_dispatches_preparation_by_stable_solver_name():
    builder = FakeBuilder([])
    registry = SolverRegistry()
    registry.register("tf-1d", builder)
    spec = SimulationSpec("tf-1d", {"grid": {"nx": 8}})

    prepared = registry.prepare(spec, key="key-42")

    assert builder.calls == [(spec, "key-42")]
    assert prepared.inputs == {"grid": {"nx": 8}}
    assert prepared.capabilities.differentiable is True


def test_lazy_registration_loads_once_and_names_do_not_trigger_loading():
    calls = []
    builder = FakeBuilder([])
    registry = SolverRegistry()

    def load_builder():
        calls.append("loaded")
        return builder

    registry.register_lazy("pic-1d", load_builder)

    assert registry.names() == ("pic-1d",)
    assert calls == []
    assert registry.resolve("pic-1d") is builder
    assert registry.resolve("pic-1d") is builder
    assert calls == ["loaded"]


def test_registration_collision_is_actionable():
    registry = SolverRegistry()
    registry.register("tf-1d", FakeBuilder([]))

    with pytest.raises(SolverAlreadyRegisteredError, match="already registered"):
        registry.register_lazy("tf-1d", lambda: FakeBuilder([]))


def test_unknown_solver_lists_available_names():
    registry = SolverRegistry()
    registry.register("tf-1d", FakeBuilder([]))

    with pytest.raises(UnknownSolverError, match=r"Unknown solver 'pic-1d'.*tf-1d"):
        registry.resolve("pic-1d")


@pytest.mark.parametrize("name", ["TF-1D", "tf_1d", "tf 1d", "-tf-1d", ""])
def test_registry_rejects_unstable_names(name):
    with pytest.raises(InvalidSolverNameError, match="lowercase kebab-case"):
        SolverRegistry().register(name, FakeBuilder([]))


def test_registry_rejects_objects_without_prepare():
    with pytest.raises(TypeError, match=r"prepare\(spec, \*, key\)"):
        SolverRegistry().register("tf-1d", object())


def test_unregister_requires_an_existing_name():
    registry = SolverRegistry()

    with pytest.raises(UnknownSolverError, match="Registered solvers: <none>"):
        registry.unregister("tf-1d")
