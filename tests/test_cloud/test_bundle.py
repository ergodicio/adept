"""Bundling and submission contract for adept.cloud.

These are the properties the runtime depends on and that a rewrite is likely to break: the archive
must be reproducible or content addressing never hits, the lock must ship even when the repo
gitignores it, committed outputs must not ride along, and an array size must reach Batch.
"""

import subprocess
import tarfile
from pathlib import Path

import pytest

from adept import cloud


def _repo(tmp_path: Path, *, lock: str = 'name = "x"\n', gitignore: str = "") -> Path:
    repo = tmp_path / "demo-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "run.py").write_text("print('hello')\n")
    (repo / "pyproject.toml").write_text('[project]\nname = "demo"\n')
    (repo / "uv.lock").write_text(lock)
    if gitignore:
        (repo / ".gitignore").write_text(gitignore)
    return repo


def _members(blob: bytes) -> set[str]:
    import gzip
    import io

    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(blob))) as tar:
        return set(tar.getnames())


def test_bundle_is_reproducible(tmp_path):
    """Same tree, same digest -- otherwise every submit re-uploads and the cache is pointless."""
    repo = _repo(tmp_path)
    first = cloud.bundle(repo)
    second = cloud.bundle(repo)
    assert first.digest == second.digest
    assert first.key == f"demo-repo/{first.digest}.tar.gz"


def test_lock_ships_even_when_gitignored(tmp_path):
    """kinetic-srs gitignores uv.lock; a .gitignore-respecting bundler would ship no lock at all."""
    repo = _repo(tmp_path, gitignore="uv.lock\n")
    assert "uv.lock" in _members(cloud.bundle(repo).blob)


def test_missing_lock_is_an_error(tmp_path):
    repo = _repo(tmp_path)
    (repo / "uv.lock").unlink()
    with pytest.raises(FileNotFoundError, match="uv sync --frozen"):
        cloud.bundle(repo)


def test_committed_outputs_are_dropped_and_reported(tmp_path):
    """These repos commit their outputs, so .gitignore alone leaves bundles enormous."""
    repo = _repo(tmp_path)
    (repo / "figure.png").write_bytes(b"\x89PNG" + b"0" * 4096)
    (repo / "fields.nc").write_bytes(b"0" * 8192)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)

    bundle = cloud.bundle(repo)
    assert "figure.png" not in _members(bundle.blob)
    assert "fields.nc" not in _members(bundle.blob)
    assert sum(count for count, _ in bundle.dropped.values()) == 2

    kept = cloud.bundle(repo, include_suffixes=[".nc"])
    assert "fields.nc" in _members(kept.blob)
    assert "figure.png" not in _members(kept.blob)


def test_exclude_glob(tmp_path):
    repo = _repo(tmp_path)
    (repo / "sims").mkdir()
    (repo / "sims" / "big.txt").write_text("x" * 100)
    assert "sims/big.txt" in _members(cloud.bundle(repo).blob)
    assert "sims/big.txt" not in _members(cloud.bundle(repo, exclude=["sims/*"]).blob)


def test_extra_files_ride_along_and_change_the_digest(tmp_path):
    """How a caller ships something generated at submit time -- a scan manifest, a resolved config."""
    repo = _repo(tmp_path)
    plain = cloud.bundle(repo)
    with_manifest = cloud.bundle(repo, extra_files={"scans/abc/manifest.json": '{"n": 3}'})

    assert "scans/abc/manifest.json" in _members(with_manifest.blob)
    assert with_manifest.digest != plain.digest
    # Same manifest, same digest: a resubmission of identical work must not re-upload.
    assert with_manifest.digest == cloud.bundle(repo, extra_files={"scans/abc/manifest.json": '{"n": 3}'}).digest


def test_untracked_edits_are_carried(tmp_path):
    """Iteration consists of uncommitted edits; a git-archive-style bundler would miss them."""
    repo = _repo(tmp_path)
    (repo / "scratch.py").write_text("x = 1\n")
    assert "scratch.py" in _members(cloud.bundle(repo).blob)


def test_venv_is_left_out(tmp_path):
    repo = _repo(tmp_path)
    (repo / ".venv" / "lib").mkdir(parents=True)
    (repo / ".venv" / "lib" / "thing.py").write_text("x = 1\n")
    assert not any(name.startswith(".venv") for name in _members(cloud.bundle(repo).blob))


def test_a_worktree_keys_under_the_main_repo(tmp_path):
    """Iterating in a worktree is normal; its directory name is a branch nickname, not a project."""
    repo = _repo(tmp_path)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"], cwd=repo, check=True)
    tree = tmp_path / "wt-some-branch"
    subprocess.run(["git", "worktree", "add", "-q", "-b", "topic", str(tree)], cwd=repo, check=True)

    assert cloud.bundle(tree).prefix == "demo-repo"
    assert cloud.bundle(tree).key.startswith("demo-repo/")
    assert cloud.bundle(tree, prefix="explicit").prefix == "explicit"


def test_not_a_git_repo(tmp_path):
    (tmp_path / "plain").mkdir()
    with pytest.raises(ValueError, match="not a git repository"):
        cloud.bundle(tmp_path / "plain")


_ADEPT_SHA = "921d5766875134830ca7db24baf21ec33a0fe446"
_ADEPT_LOCK = f'source = {{ git = "https://github.com/ergodicio/adept.git?rev=main#{_ADEPT_SHA}" }}'


@pytest.mark.parametrize(
    "lock,expected",
    [
        (_ADEPT_LOCK, _ADEPT_SHA),
        ('source = { registry = "https://pypi.org/simple" }', None),
    ],
)
def test_adept_sha_from_lock(lock, expected):
    """Both sim repos pin adept at @main, so the lock is the only record of what actually ran."""
    assert cloud.adept_sha_from_lock(lock) == expected


class _FakeS3:
    def __init__(self, present=False):
        self.present = present
        self.puts = []

    def head_object(self, **kwargs):
        if self.present:
            return {}
        from botocore.exceptions import ClientError

        raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

    def put_object(self, **kwargs):
        self.puts.append(kwargs)


class _FakeBatch:
    def __init__(self):
        self.requests = []

    def submit_job(self, **kwargs):
        self.requests.append(kwargs)
        return {"jobId": "fake-job-id", "jobName": kwargs["jobName"]}


def test_submit_passes_array_size_and_parameters(tmp_path):
    repo = _repo(tmp_path, lock=_ADEPT_LOCK)
    s3, batch = _FakeS3(), _FakeBatch()
    result = cloud.submit(
        cmd="python lpi-scan.py --array-index ${AWS_BATCH_JOB_ARRAY_INDEX:-0}",
        queue="gpu",
        repo=repo,
        bucket="code-bucket",
        extras="gpu",
        array=48,
        quiet=True,
        s3=s3,
        batch=batch,
    )

    (request,) = batch.requests
    assert request["jobQueue"] == "gpu"
    assert request["jobDefinition"] == "sim-gpu"  # queue's default resource shape
    assert request["arrayProperties"] == {"size": 48}
    assert request["parameters"]["extras"] == "gpu"
    assert request["parameters"]["code_uri"] == f"s3://code-bucket/demo-repo/{result.digest}.tar.gz"
    assert result.job_id == "fake-job-id"
    assert result.array_size == 48
    assert result.adept_sha == "921d5766875134830ca7db24baf21ec33a0fe446"
    assert result.uploaded and len(s3.puts) == 1


def test_submit_single_job_has_no_array_properties(tmp_path):
    """Batch rejects an array of size 1, so array=1 must degrade to a plain job."""
    batch = _FakeBatch()
    cloud.submit(
        cmd="python run.py",
        queue="gpu-48g",
        repo=_repo(tmp_path),
        bucket="b",
        array=1,
        quiet=True,
        s3=_FakeS3(),
        batch=batch,
    )
    assert "arrayProperties" not in batch.requests[0]


def test_identical_bundle_is_not_reuploaded(tmp_path):
    s3 = _FakeS3(present=True)
    result = cloud.submit(
        cmd="python run.py",
        queue="cpu",
        repo=_repo(tmp_path),
        bucket="b",
        quiet=True,
        s3=s3,
        batch=_FakeBatch(),
    )
    assert not result.uploaded and s3.puts == []


def test_dry_run_touches_nothing(tmp_path):
    s3, batch = _FakeS3(), _FakeBatch()
    result = cloud.submit(
        cmd="python run.py",
        queue="gpu",
        repo=_repo(tmp_path),
        bucket="b",
        dry_run=True,
        quiet=True,
        s3=s3,
        batch=batch,
    )
    assert result.job_id is None
    assert s3.puts == [] and batch.requests == []


def test_bucket_is_required(tmp_path, monkeypatch):
    monkeypatch.delenv("SIM_CODE_BUCKET", raising=False)
    with pytest.raises(ValueError, match="SIM_CODE_BUCKET"):
        cloud.submit(cmd="python run.py", queue="gpu", repo=_repo(tmp_path), quiet=True)


def test_array_index_defaults_outside_an_array_job(monkeypatch):
    monkeypatch.delenv("AWS_BATCH_JOB_ARRAY_INDEX", raising=False)
    assert cloud.array_index() == 0
    monkeypatch.setenv("AWS_BATCH_JOB_ARRAY_INDEX", "7")
    assert cloud.array_index() == 7
