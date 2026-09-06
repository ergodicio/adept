from __future__ import annotations

import json

import numpy as np
import pytest

from adept.core import (
    CheckpointCompatibility,
    CheckpointCorruptionError,
    CheckpointMetadata,
    CheckpointPreflightError,
    CheckpointStore,
    IncompatibleCheckpointError,
    LocalCheckpointStore,
    NullCheckpointStore,
    RunManifest,
    UnsupportedCheckpointVersionError,
)


def _metadata(checkpoint_id: str, *, step: int) -> CheckpointMetadata:
    return CheckpointMetadata(
        checkpoint_id=checkpoint_id,
        program="counter",
        program_version="1",
        config_fingerprint="sha256:config",
        provenance_fingerprint="sha256:provenance",
        structural_fingerprint="sha256:structure",
        simulation_time=step * 0.1,
        step=step,
        chunk_id=f"chunk-{step}",
        code={"adept": "abc123"},
        dependencies={"jax": "1.2.3"},
    )


def _advance(state, steps):
    advanced = {"position": np.array(state["position"], copy=True), "velocity": state["velocity"]}
    for _ in range(steps):
        advanced["position"] += advanced["velocity"]
    return advanced


def test_metadata_round_trips_and_rejects_unsupported_versions():
    metadata = _metadata("step-4", step=4)

    restored = CheckpointMetadata.from_dict(json.loads(json.dumps(metadata.to_dict())))

    assert restored == metadata
    assert restored.code == {"adept": "abc123"}
    assert restored.dependencies == {"jax": "1.2.3"}

    serialized = metadata.to_dict()
    serialized["schema_version"] = "999"
    with pytest.raises(UnsupportedCheckpointVersionError, match="unsupported checkpoint schema"):
        CheckpointMetadata.from_dict(serialized)


def test_metadata_derives_reproducibility_fingerprints_from_a_manifest():
    manifest = RunManifest(
        raw_config={"grid": {"nx": 8}},
        resolved_config={"grid": {"nx": 8, "dx": 0.25}},
        units={"time": "second"},
        seed=42,
        key_provenance="jax-key:00000000:0000002a",
        code={"adept": "abc123"},
        dependencies={"jax": "1.2.3"},
        structural_fingerprint="sha256:structure",
    )

    metadata = CheckpointMetadata.from_manifest(
        manifest,
        checkpoint_id="step-4",
        program="counter",
        program_version="1",
        simulation_time=0.4,
        step=4,
    )

    assert metadata.config_fingerprint.startswith("sha256:")
    assert metadata.provenance_fingerprint.startswith("sha256:")
    assert metadata.structural_fingerprint == manifest.structural_fingerprint
    assert metadata.code == manifest.code
    assert metadata.dependencies == manifest.dependencies


def test_null_store_is_an_explicit_noop():
    store = NullCheckpointStore()

    assert isinstance(store, CheckpointStore)
    assert store.save({"state": 1}, _metadata("unused", step=0)).wait() is None
    assert store.list() == ()
    assert store.latest() is None
    with pytest.raises(LookupError, match="no checkpoints"):
        store.restore("missing", {"state": 0})


def test_local_store_round_trips_state_and_records_tree_metadata(tmp_path):
    store = LocalCheckpointStore(tmp_path / "checkpoints")
    state = {
        "position": np.arange(6, dtype=np.float64).reshape(2, 3),
        "step": np.array(4, dtype=np.int32),
    }

    pending = store.save(state, _metadata("step-4", step=4))
    reference = pending.wait()

    assert reference is not None
    assert reference.checkpoint_id == "step-4"
    assert [leaf.path for leaf in reference.metadata.leaves] == ["['position']", "['step']"]
    assert [leaf.shape for leaf in reference.metadata.leaves] == [(2, 3), ()]
    assert [leaf.dtype for leaf in reference.metadata.leaves] == ["float64", "int32"]
    assert [item.checkpoint_id for item in store.list()] == ["step-4"]
    assert store.latest() == reference

    target = {"position": np.zeros((2, 3), dtype=np.float64), "step": np.array(0, dtype=np.int32)}
    restored = store.restore(
        reference,
        target,
        compatibility=CheckpointCompatibility.from_metadata(reference.metadata),
    )

    np.testing.assert_array_equal(restored["position"], state["position"])
    np.testing.assert_array_equal(restored["step"], state["step"])


def test_interrupted_serial_run_resumes_to_uninterrupted_result(tmp_path):
    store = LocalCheckpointStore(tmp_path / "checkpoints")
    initial = {"position": np.array([0.0, 1.0]), "velocity": np.array([0.25, -0.5])}
    uninterrupted = _advance(initial, 10)
    interrupted = _advance(initial, 4)

    saved = store.save(interrupted, _metadata("step-4", step=4)).wait()
    assert saved is not None
    restored = store.restore(
        saved,
        {"position": np.zeros(2), "velocity": np.zeros(2)},
        compatibility=CheckpointCompatibility.from_metadata(saved.metadata),
    )
    resumed = _advance(restored, 6)

    np.testing.assert_allclose(resumed["position"], uninterrupted["position"])
    np.testing.assert_allclose(resumed["velocity"], uninterrupted["velocity"])


def test_restore_places_arrays_like_the_jax_target(tmp_path):
    import jax
    import jax.numpy as jnp

    store = LocalCheckpointStore(tmp_path / "checkpoints")
    reference = store.save({"value": jnp.arange(3, dtype=jnp.float32)}, _metadata("step-1", step=1)).wait()
    assert reference is not None

    target = {"value": jax.device_put(jnp.zeros(3, dtype=jnp.float32))}
    restored = store.restore(reference, target)

    assert isinstance(restored["value"], jax.Array)
    assert restored["value"].sharding == target["value"].sharding
    np.testing.assert_array_equal(restored["value"], np.arange(3, dtype=np.float32))


def test_failed_save_keeps_prior_latest_and_excludes_partial_checkpoint(tmp_path, monkeypatch):
    store = LocalCheckpointStore(tmp_path / "checkpoints")
    first = store.save({"value": np.array([1.0])}, _metadata("step-1", step=1)).wait()
    original_write = store._write_arrays

    def fail_after_partial_write(path, arrays):
        original_write(path, arrays)
        raise RuntimeError("injected save failure")

    monkeypatch.setattr(store, "_write_arrays", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="injected save failure"):
        store.save({"value": np.array([2.0])}, _metadata("step-2", step=2))

    assert first is not None
    assert store.latest() == first
    assert [item.checkpoint_id for item in store.list()] == ["step-1"]
    assert not (store.root / "step-2").exists()


def test_latest_update_failure_rolls_back_commit_and_allows_retry(tmp_path, monkeypatch):
    store = LocalCheckpointStore(tmp_path / "checkpoints")
    first = store.save({"value": np.array([1.0])}, _metadata("step-1", step=1)).wait()
    original_write_latest = store._write_latest

    def fail_latest_update(checkpoint_id):
        original_write_latest(checkpoint_id)
        raise RuntimeError(f"injected latest failure for {checkpoint_id}")

    monkeypatch.setattr(store, "_write_latest", fail_latest_update)
    with pytest.raises(RuntimeError, match="injected latest failure"):
        store.save({"value": np.array([2.0])}, _metadata("step-2", step=2))

    assert first is not None
    assert store.latest() == first
    assert [item.checkpoint_id for item in store.list()] == ["step-1"]
    assert not (store.root / "step-2").exists()

    monkeypatch.setattr(store, "_write_latest", original_write_latest)
    retried = store.save({"value": np.array([2.0])}, _metadata("step-2", step=2)).wait()
    assert retried is not None
    assert store.latest() == retried


def test_restore_rejects_incompatible_programs_and_state_trees(tmp_path):
    store = LocalCheckpointStore(tmp_path / "checkpoints")
    reference = store.save({"value": np.ones(2)}, _metadata("step-1", step=1)).wait()
    assert reference is not None

    incompatible_program = CheckpointCompatibility(
        program="other-program",
        program_version="1",
        config_fingerprint="sha256:config",
        structural_fingerprint="sha256:structure",
    )
    with pytest.raises(IncompatibleCheckpointError, match="program"):
        store.restore(reference, {"value": np.zeros(2)}, compatibility=incompatible_program)

    with pytest.raises(IncompatibleCheckpointError, match="shape"):
        store.restore(reference, {"value": np.zeros(3)})


def test_validate_detects_corrupted_state(tmp_path):
    store = LocalCheckpointStore(tmp_path / "checkpoints")
    reference = store.save({"value": np.array([1.0])}, _metadata("step-1", step=1)).wait()
    assert reference is not None
    np.savez(store.root / "step-1" / "state.npz", leaf_000000=np.array([9.0]))

    with pytest.raises(CheckpointCorruptionError, match="checksum"):
        store.validate(reference)


def test_preflight_reports_an_unusable_root(tmp_path):
    root = tmp_path / "not-a-directory"
    root.write_text("occupied", encoding="utf-8")

    with pytest.raises(CheckpointPreflightError, match="not writable"):
        LocalCheckpointStore(root).preflight()


def test_checkpoint_ids_cannot_collide_with_the_latest_pointer():
    with pytest.raises(ValueError, match="reserved"):
        _metadata("latest.json", step=0)
