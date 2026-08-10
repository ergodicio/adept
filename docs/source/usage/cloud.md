# Running on AWS Batch

`adept.cloud` submits a working tree to AWS Batch. The Batch sim images (`sim-cpu`, `sim-gpu`)
contain the environment only — **no application code**. Code arrives at run time as a
content-addressed tarball in S3, so editing a line of physics costs an upload of a few hundred KB
instead of a `docker build`, an ECR push and a `cdk deploy`.

The runtime half of the contract is `continuum-infra/sim-runner/bootstrap.sh`, which downloads the
bundle, runs `uv sync --frozen` against the lock inside it, and executes the submitted command.

## Submitting

```python
from adept.cloud import submit

result = submit(
    cmd="python run.py --cfg configs/vlasov-1d/epw",
    queue="gpu",
    extras="gpu",
)
print(result.job_id, result.adept_sha)
```

or from the command line:

```bash
uv run python -m adept.cloud --cmd 'python run.py --cfg configs/vlasov-1d/epw' --queue gpu --extras gpu
```

```{note}
The `--cmd` string runs *inside the Batch container*, after `bootstrap.sh` has done its own
`uv sync --frozen`. It is not a local command, so it does not follow this repo's `uv run`
convention.
```

`--dry-run` bundles and reports without uploading or submitting — the cheapest way to see what a
bundle would contain.

`cmd` is a command rather than an `(entry, config)` pair because the repos that submit disagree
about their CLIs. The bucket comes from `bucket=` or `$SIM_CODE_BUCKET`, and `job_definition`
defaults to the generic definition matching the queue's resource shape.

## Queues

| Queue | Shape | Pricing |
|---|---|---|
| `gpu` | 1 × 24 GB (L4 / A10G) | spot |
| `gpu-48g` | 1 × 48 GB (L40S) | on-demand |
| `cpu` | 2 vCPU | spot |
| `cpu-hmem` | 8 vCPU / 60 GB | spot |

## Scans are array jobs

Each sim is single-GPU, so an N-point scan is **one** array job of size N rather than N
submissions — with Batch's spot retries for free:

```python
result = submit(cmd="python scan.py --index ${AWS_BATCH_JOB_ARRAY_INDEX:-0}", queue="gpu", array=48)
```

`$AWS_BATCH_JOB_ARRAY_INDEX` survives Batch's `Ref::` substitution unexpanded and is expanded by
`bootstrap.sh` inside the container. In the worker, read it with `adept.cloud.array_index()`, which
returns 0 when the process is not an array member, so the same entry point runs locally by hand.

An `array` of 0 or 1 submits a plain job (Batch has no array of size 1), in which case the variable
is unset — hence the `${...:-0}` default above.

## What gets bundled

Bundling uses `git ls-files -co --exclude-standard`: tracked files plus untracked ones that
`.gitignore` does not exclude. That **carries uncommitted edits**, which is what iteration
consists of, while leaving `.venv`, artifacts and `mlflow.db` out for free.

On top of that:

- `uv.lock` and `pyproject.toml` ship regardless of git status. Some repos gitignore their lock,
  and the runner installs from it.
- Derived formats (`.png`, `.nc`, `.h5`, `.npz`, …) are dropped by suffix, because these repos
  commit their outputs. Nothing is dropped silently — the count and size are reported on every
  submit, and `include_suffixes=[".npy"]` puts one back when a run needs it as *input*.
- `exclude=["sims/*"]` drops a path glob.
- `extra_files={"scans/abc/manifest.json": ...}` adds content generated at submit time that is not
  in the tree at all. It participates in the digest.

The archive is byte-reproducible: the tar is built uncompressed and gzipped with `mtime=0`, and
member metadata is normalised, so the same tree always yields the same digest and an unchanged
bundle is never re-uploaded.

## Which adept actually ran

Repos that depend on adept typically declare `adept @ git+https://github.com/ergodicio/adept@main`,
i.e. unpinned. `submit()` resolves the lock inside the bundle and returns the adept commit the job
will install as `result.adept_sha`. Record it alongside the Batch job id — it is the only record of
the physics that ran.
