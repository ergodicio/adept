"""Rebuild paired ADEPT/CUDA timing comparisons from the preserved raw reports."""

import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parent
pairs = []
for nx, ranks in [(134400, 4), (134400, 8), (268800, 8)]:
    reference = json.loads((root / f"jax-nx{nx}-g{ranks}.json").read_text())
    jax_rows = reference["rows"]
    cuda_rows = [json.loads(line) for line in (root / f"cuda-nx{nx}-g{ranks}.jsonl").read_text().splitlines()]
    assert len(jax_rows) == len(cuda_rows) == 3
    assert reference["adept_commit"] == "ddd70b957dc3df85f0b5ec84af927c25d09d24e0"
    assert reference["libraries"]["jax"] == "0.10.2"
    assert len(reference["hosts"]) == ranks and len(set(reference["hosts"])) == ranks // 4
    for row in [*jax_rows, *cuda_rows]:
        assert row["nx"] == nx and row["nv"] == 4096 and row["ranks"] == ranks
        assert row["steps"] == 10 and row["warmup"] == 3
    for row in cuda_rows:
        assert row["finite"] and row["nu"] == 1e-12 and row["integrator"] == "sixth"
        assert row["velocity"] == "cubic" and row["layout"] == "halos" and row["cell_centred"]
        assert row["dt"] == 0.02 and row["length"] == 12600 and row["vmax"] == 15
        for host in set(reference["hosts"]):
            assert f"host={host} " in row["rank_log"]
        jax_mass = jax_rows[0]["sum"] * (12600 / nx) * (30 / 4096)
        assert math.isclose(jax_mass, row["final_mass"], rel_tol=3e-10)
        assert math.isclose(jax_rows[0]["minimum"], row["minimum_f"], rel_tol=1e-10, abs_tol=1e-11)
    a = statistics.median(row["step_ms"] for row in jax_rows)
    b = statistics.median(row["step_ms"] for row in cuda_rows)
    pairs.append(
        dict(
            nx=nx,
            nv=4096,
            nodes=ranks // 4,
            gpus=ranks,
            adept_jax_ms=a,
            native_cuda_ms=b,
            speedup=a / b,
            adept_jax_ms_range=[min(row["step_ms"] for row in jax_rows), max(row["step_ms"] for row in jax_rows)],
            native_cuda_ms_range=[min(row["step_ms"] for row in cuda_rows), max(row["step_ms"] for row in cuda_rows)],
            hosts=sorted(set(reference["hosts"])),
            repeats=3,
            native_max_allocated_bytes_per_rank=max(row["max_allocated_bytes_per_rank"] for row in cuda_rows),
            endpoint_aggregate_check="mass and minimum agree; pointwise validation is reported separately",
        )
    )
archive = root / "source.tar.gz"
with tarfile.open(archive) as snapshot:
    for line in (root / "source.sha256").read_text().splitlines():
        expected, name = line.split()
        assert hashlib.sha256(snapshot.extractfile(name).read()).hexdigest() == expected
checks = json.loads((root / "integration-checks.json").read_text())
assert len(checks) == 7 and all(row["passed"] for row in checks)
report = dict(
    date="2026-10-24",
    system="NERSC Perlmutter",
    allocation="60221398",
    device="NVIDIA A100-SXM4-40GB",
    precision="float64",
    dt=0.02,
    nu=1e-12,
    libraries=reference["libraries"],
    adept_commit=reference["adept_commit"],
    timing=(
        "Median of 3 repeats, 10 steps after 3 warmup steps, maximum rank elapsed time; "
        "no concurrent benchmark workloads. Both paths reset to identical t600 input."
    ),
    scope=(
        "Steady-state physics operators. Excludes compilation, preprocessing, save callbacks, I/O and "
        "postprocessing. Not a full-run wall-clock speedup or a full t0-to-2000 reproduction."
    ),
    comparison=pairs,
    integration_checks=checks,
    local_tests=dict(native=5, existing_regressions=33),
    source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
)
(root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(pairs, indent=2))
