"""Command line for the parity harness.

    python -m adept._lpse2d.parity run   <deck|lpse.parms> [--windows 0.3-0.9,0.9-1.5] [--overrides '{...}'] [--out DIR]
    python -m adept._lpse2d.parity compare <series.xr|series.nc> <lpse.metrics> --windows ...
    python -m adept._lpse2d.parity log   <run_id> <series.xr|series.nc> <lpse run dir> --windows ... [--full]
    python -m adept._lpse2d.parity fetch <deck> [--dest DIR]
    python -m adept._lpse2d.parity verify <run_id> [--windows ...] [--rtol 1e-6]

``run`` translates the deck (a shipped-deck name resolves through ``original-lpse``), runs it
through ``ergoExo`` and — when the deck's reference run is found — logs the comparison to the
new MLflow run. ``verify`` downloads a finished cross-check run's ``binary/series.xr`` and
``lpse_reference/lpse.metrics`` and checks that ``compare`` reproduces the ratio metrics
stored on that run (the acceptance check for moving this tooling into the repository).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import harness, reference


def _windows_for(deck: str, spec: str | None):
    if spec:
        return harness.parse_windows(spec)
    if deck in harness.DEFAULT_WINDOWS:
        return harness.DEFAULT_WINDOWS[deck]
    raise SystemExit(f"no default fit windows for {deck!r}; pass --windows lo-hi[,lo-hi]")


def _load_overrides(spec: str | None) -> dict:
    if not spec:
        return {}
    if Path(spec).is_file():
        import yaml

        with open(spec) as fi:
            return yaml.safe_load(fi) or {}
    return json.loads(spec)


def cmd_run(args) -> int:
    parms = Path(args.deck)
    if not parms.is_file():
        found = reference.deck_path(args.deck)
        if found is None:
            raise SystemExit(f"{args.deck}: not a file and not a shipped deck under {reference.lpse_root()}")
        parms = found
    deck = parms.resolve().parent.name
    overrides = harness.merge_overrides(dict(harness.DEFAULT_OVERRIDES.get(deck, {})), _load_overrides(args.overrides))
    result = harness.run_deck(parms, overrides, out_dir=args.out, run=args.run, experiment=args.experiment)
    print(f"mlflow run {result.run_id}; unsupported: {result.report['unsupported']}")
    lpse_dir = reference.reference_run_dir(deck, download=not args.no_download)
    if lpse_dir is None:
        print(f"no LPSE reference run for {deck}; comparison skipped")
        return 0
    metrics = harness.compare(result.series, lpse_dir / "data" / "lpse.metrics", _windows_for(deck, args.windows))
    _print_metrics(metrics)
    if result.run_id and not args.no_log:
        harness.log_reference(result.run_id, lpse_dir, metrics, deck=deck, full=args.full)
    return 0


def _print_metrics(metrics: dict[str, float]):
    for k, v in metrics.items():
        print(f"  {k}: {v:.6g}")


def cmd_compare(args) -> int:
    metrics = harness.compare(args.series, args.metrics, harness.parse_windows(args.windows))
    _print_metrics(metrics)
    return 0


def cmd_log(args) -> int:
    lpse_dir = Path(args.lpse_dir)
    metrics = harness.compare(args.series, lpse_dir / "data" / "lpse.metrics", harness.parse_windows(args.windows))
    _print_metrics(metrics)
    harness.log_reference(args.run_id, lpse_dir, metrics, deck=args.deck, full=args.full)
    return 0


def cmd_fetch(args) -> int:
    dest = reference.download_reference(args.deck, args.dest, experiment=args.experiment)
    print(f"reference {args.deck} -> {dest}")
    return 0


def cmd_verify(args) -> int:
    from adept import patched_mlflow as mlflow

    run = mlflow.MlflowClient().get_run(args.run_id)
    deck = run.data.tags.get("lpse_reference_deck", run.info.run_name)
    dest = Path(args.dest) if args.dest else reference.reference_cache_dir() / "verify" / args.run_id
    reference.download_run_artifacts(args.run_id, dest / "binary", "binary/series.xr")
    reference.download_run_artifacts(args.run_id, dest / "lpse_reference", "lpse_reference/lpse.metrics")
    stored = {
        k: v for k, v in run.data.metrics.items() if "_energy_growth_" in k or k.startswith(("epw_half_max", "iaw_max"))
    }
    computed = harness.compare(
        dest / "binary" / "series.xr", dest / "lpse_reference" / "lpse.metrics", _windows_for(deck, args.windows)
    )
    worst = 0.0
    missing = []
    for k, v in sorted(stored.items()):
        if k not in computed:
            missing.append(k)
            continue
        c = computed[k]
        err = abs(c - v) / max(abs(v), 1e-300)
        worst = max(worst, err)
        flag = "" if err <= args.rtol else "   <-- MISMATCH"
        print(f"  {k}: stored {v:.10g}  recomputed {c:.10g}  rel {err:.2e}{flag}")
    extra = sorted(set(computed) - set(stored))
    print(f"{deck} ({args.run_id}): {len(stored)} stored metrics, worst relative difference {worst:.2e}")
    if missing:
        print("stored but not recomputed:", missing)
    if extra:
        print("recomputed but not stored:", extra)
    ok = worst <= args.rtol and not missing
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m adept._lpse2d.parity", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="translate + run a deck, compare and log against its reference")
    p.add_argument("deck", help="path to lpse.parms or a shipped deck name (test_006)")
    p.add_argument("--windows", help="fit windows in ps, lo-hi[,lo-hi]; default from DEFAULT_WINDOWS")
    p.add_argument("--overrides", help="JSON string or YAML file of config overrides")
    p.add_argument("--out", help="directory for config.yaml / series.nc / fields.nc / metrics.npy")
    p.add_argument("--run", help="MLflow run name (default: deck directory name)")
    p.add_argument("--experiment", default=reference.LPSE_PARITY_EXPERIMENT)
    p.add_argument("--full", action="store_true", help="log the whole reference run tree, not just metrics/parms")
    p.add_argument("--no-log", action="store_true", help="compare but do not log the comparison to MLflow")
    p.add_argument("--no-download", action="store_true", help="do not fetch a missing reference from MLflow")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", help="growth-rate comparison of an adept series with an LPSE metrics table")
    p.add_argument("series")
    p.add_argument("metrics")
    p.add_argument("--windows", required=True)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("log", help="attach an LPSE reference + comparison metrics to an existing adept run")
    p.add_argument("run_id")
    p.add_argument("series")
    p.add_argument("lpse_dir", help="reference run directory (contains lpse.parms and data/)")
    p.add_argument("--windows", required=True)
    p.add_argument("--deck")
    p.add_argument("--full", action="store_true")
    p.set_defaults(func=cmd_log)

    p = sub.add_parser("fetch", help="download a deck's reference run from MLflow")
    p.add_argument("deck")
    p.add_argument("--dest")
    p.add_argument("--experiment", default=reference.LPSE_PARITY_EXPERIMENT)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("verify", help="recompute a cross-check run's stored ratio metrics from its artifacts")
    p.add_argument("run_id")
    p.add_argument("--windows")
    p.add_argument("--dest")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.set_defaults(func=cmd_verify)

    args = ap.parse_args(argv)
    np.set_printoptions(precision=6)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
