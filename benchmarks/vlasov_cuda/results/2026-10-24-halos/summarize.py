"""Rebuild the compact report from the adjacent preserved measurements."""

import hashlib
import json
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parent
groups = defaultdict(list)
for nx in [134400, 268800]:
    for layout in ["transpose", "halos"]:
        for line in (root / f"nx{nx}-{layout}.jsonl").read_text().splitlines():
            row = json.loads(line)
            groups[nx, row["nodes"], row["ranks"], layout].append(row)
timings = []
for (nx, nodes, ranks, layout), rows in sorted(groups.items()):
    assert len(rows) == 3 and all(
        row["finite"]
        and row["steps"] == 10
        and row["warmup"] == 3
        and row["integrator"] == "sixth"
        and row["nu"] == 1e-12
        and row["velocity"] == "cubic"
        and row["cell_centred"]
        for row in rows
    )
    result = dict(nx=nx, nv=4096, nodes=nodes, gpus=ranks, layout=layout, repeats=len(rows))
    for key in [
        "step_ms",
        "stream_ms",
        "spline_ms",
        "pack_ms",
        "mpi_ms",
        "field_ms",
        "collision_ms",
        "max_allocated_bytes_per_rank",
    ]:
        result[key] = statistics.median(row[key] for row in rows)
    result["step_ms_range"] = [min(row["step_ms"] for row in rows), max(row["step_ms"] for row in rows)]
    result["maximum_halo_width"] = max(row["max_halo_width"] for row in rows)
    result["sent_bytes_per_step_max_rank"] = statistics.median(
        (row["slab_sent_bytes_per_rank"] + row["halo_sent_bytes_per_rank_max"]) / row["steps"] for row in rows
    )
    result["max_abs_relative_mass_change"] = max(abs(row["mass_relative_change"]) for row in rows)
    timings.append(result)
pairs = []
for nx in [134400, 268800]:
    for nodes, ranks in [(1, 4), (2, 8)]:
        match = {row["layout"]: row for row in timings if (row["nx"], row["nodes"], row["gpus"]) == (nx, nodes, ranks)}
        original, halo = match["transpose"], match["halos"]
        pairs.append(
            dict(
                nx=nx,
                nodes=nodes,
                gpus=ranks,
                transpose_ms=original["step_ms"],
                halo_ms=halo["step_ms"],
                speedup=original["step_ms"] / halo["step_ms"],
                sent_bytes_reduction=1
                - halo["sent_bytes_per_step_max_rank"] / original["sent_bytes_per_step_max_rank"],
            )
        )
checks = []
for name in ["comparison-nx134400-smoke.json", "comparison-nx134400-window.json", "comparison-nx268800-window.json"]:
    rows = json.loads((root / name).read_text())
    assert rows and all(row["passed"] for row in rows)
    checks.extend(rows)
validation = []
for name in ["validation-local-cpu.json", "validation-local-cuda.json", "validation-natural-cpu.json"]:
    rows = json.loads((root / name).read_text())
    assert all(row["passed"] for row in rows)
    validation.append(dict(report=name, checks=len(rows), max_abs_error=max(row["max_abs_error"] for row in rows)))
archive = root / "source.tar.gz"
# These measurements refer to the preserved source snapshot, even after the
# solver is promoted into ADEPT or the working tree changes.
with tarfile.open(archive) as snapshot:
    sources = {
        Path(member.name).name: snapshot.extractfile(member).read()
        for member in snapshot.getmembers()
        if member.isfile()
        and len(Path(member.name).parts) == 2
        and (Path(member.name).suffix in [".cpp", ".hpp", ".py"] or Path(member.name).name == "CMakeLists.txt")
    }
for line in (root / "compiled-source.sha256").read_text().splitlines():
    expected, name = line.split()
    assert hashlib.sha256(sources[Path(name).name]).hexdigest() == expected
report = dict(
    date="2026-10-24",
    system="NERSC Perlmutter",
    device="NVIDIA A100-SXM4-40GB",
    allocation="60206959",
    reference_allocation="60205333",
    nodes=["nid001024", "nid001025"],
    precision="float64",
    dt=0.02,
    nu=1e-12,
    software=dict(cuda="13.2.78", gcc="14.3.0", cray_mpich="9.1.0", cmake="3.30.2"),
    reference_libraries=json.loads((root / "reference-library-versions.json").read_text()),
    integrator="sixth",
    velocity="interpax cubic C1 Hermite",
    streaming="spectral exponential",
    timing=(
        "Median of 3 repeats; 10 measured steps after 3 warmup steps; maximum rank mean; "
        "explicit GPU synchronization; no overlap. Input/output excluded."
    ),
    reference_cuda_executable_sha256="c5db5fa71e9d1119e7cb581a9e67624f42155c006da45184cb55de6542255b9a",
    measured_executable_sha256=(root / "executable.sha256").read_text().split()[0],
    comparison_scope=(
        "Full native grids, mature checkpoint t600 to t602. Not a full t0-to-2000 trajectory "
        "reproduction or a CUDA-versus-JAX speed comparison."
    ),
    checks=dict(
        native_cases=len(checks),
        max_abs_f_error=max(row["max_abs_f_error"] for row in checks),
        max_relative_l2_f_error=max(row["relative_l2_f_error"] for row in checks),
        max_relative_l2_e_error=max(row["relative_l2_e_error"] for row in checks),
        all_hole_counts_match=all(row["hole_counts_match"] for row in checks),
        validation=validation,
    ),
    paired=pairs,
    timing_rows=timings,
    source_sha256={name: hashlib.sha256(content).hexdigest() for name, content in sorted(sources.items())},
)
if archive.exists():
    report["source_archive"] = dict(file=archive.name, sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
(root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(dict(checks=report["checks"], paired=pairs), indent=2))
