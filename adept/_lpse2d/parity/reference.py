"""Locate the original-LPSE tree, its shipped decks and its reference runs.

The parity work compares adept ``lpse2d`` runs against runs of the original LPSE C++ code
(deck version 5.1.2). The C++ tree, ``original-lpse``, is a sibling repository of adept and
its reference runs live in ``<root>/runs/<deck>/`` (``lpse.parms``, ``lpse.out`` and the
``data/`` outputs). Every reference run is also archived on the MLflow experiment
``lpse-parity`` (id 189060) as the ``lpse_reference/`` artifact tree of a run tagged
``lpse_reference_deck = <deck>``; that copy is the fallback when the local tree is absent
(Perlmutter, CI with credentials).

Resolution order for the tree:

1. ``$LPSE_ROOT``;
2. a directory named ``original-lpse`` beside this repository or beside one of its
   ancestors (worktrees live under ``<repo>/.claude/worktrees/<name>``, so the walk goes up
   a bounded number of levels);
3. none — callers skip or fall back to MLflow.

Nothing here is imported by the solver; it is test and tooling support only.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

LPSE_PARITY_EXPERIMENT = "lpse-parity"
REFERENCE_ARTIFACT_DIR = "lpse_reference"
_MAX_PARENT_LEVELS = 6


def lpse_root(env: str = "LPSE_ROOT") -> Path | None:
    """The ``original-lpse`` checkout, or ``None`` when it cannot be found."""
    if os.environ.get(env):
        root = Path(os.environ[env]).expanduser()
        return root if root.is_dir() else None
    here = Path(__file__).resolve()
    for parent in list(here.parents)[: _MAX_PARENT_LEVELS + 3]:
        candidate = parent / "original-lpse"
        if (candidate / "examples").is_dir() or (candidate / "runs").is_dir():
            return candidate
    return None


def deck_path(deck: str, root: Path | None = None) -> Path | None:
    """The deck's ``lpse.parms``: the shipped example ``<root>/examples/testRuns/<deck>/``,
    else the reference run's own copy (``<root>/runs/<deck>/`` -- the homogeneous controls
    ``hom_srs_020`` / ``hom_tpd_023`` exist only there -- or the downloaded reference)."""
    root = root or lpse_root()
    candidates = []
    if root is not None:
        candidates += [root / "examples" / "testRuns" / deck / "lpse.parms", root / "runs" / deck / "lpse.parms"]
    candidates.append(reference_cache_dir() / deck / "lpse.parms")
    for path in candidates:
        if path.is_file():
            return path
    return None


def reference_cache_dir() -> Path:
    """Where downloaded reference runs are kept (``$LPSE_REFERENCE_CACHE`` or ``~/.cache``)."""
    return Path(os.environ.get("LPSE_REFERENCE_CACHE", Path.home() / ".cache" / "adept" / "lpse-reference"))


def _has_reference_outputs(run_dir: Path) -> bool:
    return (run_dir / "data" / "lpse.metrics").is_file() or (run_dir / "lpse.metrics").is_file()


def reference_run_dir(
    deck: str,
    root: Path | None = None,
    download: bool = True,
    experiment: str = LPSE_PARITY_EXPERIMENT,
) -> Path | None:
    """Directory of the LPSE reference run for ``deck`` (contains ``lpse.parms`` and ``data/``).

    Looks in ``<root>/runs/<deck>`` first, then in the download cache, then (with
    ``download``) fetches the ``lpse_reference/`` tree from MLflow. Returns ``None`` when
    none of these is available — never raises for a missing run, so tests can skip."""
    root = root or lpse_root()
    if root is not None and _has_reference_outputs(root / "runs" / deck):
        return root / "runs" / deck
    cached = reference_cache_dir() / deck
    if _has_reference_outputs(cached):
        return cached
    if not download:
        return None
    try:
        return download_reference(deck, cached, experiment=experiment)
    except Exception as exc:  # credentials, network, missing run: all mean "not available"
        logger.info("LPSE reference run %s not downloadable: %s: %s", deck, type(exc).__name__, exc)
        return None


def _mlflow_client():
    from adept import patched_mlflow as mlflow

    return mlflow.MlflowClient()


def find_reference_run(deck: str, experiment: str = LPSE_PARITY_EXPERIMENT):
    """The MLflow run holding the reference outputs of ``deck``: prefers a run tagged
    ``lpse_reference_only`` or ``lpse_reference_complete``, newest first."""
    client = _mlflow_client()
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise LookupError(f"MLflow experiment {experiment!r} not found")
    runs = client.search_runs(
        exp.experiment_id,
        filter_string=f'tags.lpse_reference_deck = "{deck}"',
        order_by=["start_time DESC"],
        max_results=20,
    )
    complete = [r for r in runs if r.data.tags.get("lpse_reference_only") or r.data.tags.get("lpse_reference_complete")]
    if not (complete or runs):
        raise LookupError(f"no run tagged lpse_reference_deck = {deck!r} in {experiment!r}")
    return (complete or runs)[0]


def _s3_location(artifact_uri: str, subdir: str) -> tuple[str, str]:
    if not artifact_uri.startswith("s3://"):
        raise ValueError(f"artifact store {artifact_uri} is not S3; download it with the MLflow client instead")
    bucket = artifact_uri.split("/")[2]
    prefix = "/".join(artifact_uri.split("/")[3:]).rstrip("/")
    return bucket, f"{prefix}/{subdir}".rstrip("/")


def download_run_artifacts(run_id: str, dest: Path, subdir: str = "") -> Path:
    """Download the ``subdir`` artifact tree (or single artifact file) of an MLflow run into
    ``dest`` with boto3 — the tracking server's own artifact proxy is too slow for field
    files. A tree keeps its layout under ``dest``; a file lands at ``dest/<basename>``.
    Returns ``dest``."""
    import boto3

    run = _mlflow_client().get_run(run_id)
    bucket, prefix = _s3_location(run.info.artifact_uri, subdir)
    s3 = boto3.client("s3")
    dest = Path(dest)
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key == prefix:
                rel = Path(key).name
            elif key.startswith(prefix + "/"):
                rel = key[len(prefix) + 1 :]
            else:
                continue  # a sibling whose name merely starts with the prefix
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(target))
            n += 1
    if n == 0:
        raise LookupError(f"run {run_id} has no artifacts under {subdir!r}")
    return dest


def download_reference(deck: str, dest: Path | None = None, experiment: str = LPSE_PARITY_EXPERIMENT) -> Path:
    """Fetch the ``lpse_reference/`` tree of ``deck`` from MLflow into ``dest`` (default: the
    cache) laid out like ``<root>/runs/<deck>/``. Needs the MLflow tracking environment and
    AWS credentials for the artifact bucket."""
    dest = Path(dest) if dest is not None else reference_cache_dir() / deck
    run = find_reference_run(deck, experiment=experiment)
    download_run_artifacts(run.info.run_id, dest, REFERENCE_ARTIFACT_DIR)
    if not _has_reference_outputs(dest):
        raise LookupError(f"run {run.info.run_id} ({deck}) has no lpse.metrics under {REFERENCE_ARTIFACT_DIR}/")
    (dest / ".mlflow_run_id").write_text(run.info.run_id + "\n")
    logger.info("downloaded LPSE reference %s from run %s to %s", deck, run.info.run_id, dest)
    return dest
