"""Offline regeneration of the WarpX canned plot set from saved NetCDFs.

``post.collect`` converts a finished run's openPMD + reduced output into
per-diagnostic NetCDFs under ``binary/`` (code units, OSIRIS contract) and
renders the canned plots. This regenerates that *same* plot set from the
``binary/`` NetCDFs alone — no rerun, no raw openPMD files — mirroring
:mod:`adept.osiris.regen`.

Usage::

    python -m adept.warpx.regen <run-or-binary-dir> [--out DIR] [options]

The source may also be a raw WarpX run directory (containing ``diags/``):
the NetCDF conversion is then performed first into ``<run>/binary/``, with
the normalization recovered from ``units.yaml`` / ``config.yaml`` /
the rendered ``inputs`` next to it (see
:func:`adept.warpx.io.code_units_for_run`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adept.warpx import io as _io


def find_binary_dir(path: str | Path) -> Path:
    path = Path(path)
    if (path / "binary").is_dir():
        return path / "binary"
    return path


def default_out_dir(src: str | Path) -> Path:
    src = Path(src)
    run_dir = src if (src / "binary").is_dir() else src.parent
    return run_dir / "plots_regen"


def load_output_cfg(src: str | Path) -> dict:
    """Best-effort read of the manifest ``output:`` block for a run."""
    src = Path(src)
    for candidate in (src / "config.yaml", src.parent / "config.yaml"):
        if candidate.is_file():
            try:
                import yaml

                cfg = yaml.safe_load(candidate.read_text()) or {}
            except Exception:  # a malformed config must not block regeneration
                return {}
            return cfg.get("output") or {}
    return {}


def regenerate(
    src: str | Path,
    out_dir: str | Path | None = None,
    *,
    use_config: bool = True,
    **overrides,
) -> dict[str, Path]:
    """Regenerate the canned plot set from a run's saved NetCDFs.

    ``src`` is a run directory (containing ``binary/`` and optionally
    ``config.yaml``), a ``binary/`` NetCDF directory, or a raw WarpX run
    directory (containing ``diags/``, which is converted first). Plot knobs
    default to the run's ``output:`` config; keywords in ``overrides``
    (``v_th``, ``omega_k_zoom``, ``dpi``, ``n_panels``) apply on top.
    """
    from adept.warpx import plots as _plots

    src = Path(src)
    binary = find_binary_dir(src)
    if not any(binary.rglob("*.nc")) and (src / "diags").is_dir():
        # Raw run dir with no conversion yet: build binary/ first.
        binary = src / "binary"
        units = _io.code_units_for_run(src)
        if units is None:
            print("[regen] no reference density found — converting in SI units")
        _io.save_run_datasets(src, binary, units=units)
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir(src)
    kwargs = _plots.canned_plot_kwargs(load_output_cfg(src) if use_config else None)
    kwargs.update(overrides)
    return _plots.save_canned_plots(binary, out_dir, **kwargs)


def _summarize(written: dict[str, Path], out_dir: Path) -> None:
    families: dict[str, int] = {}
    for name in written:
        fam = name.split("/", 1)[0]
        families[fam] = families.get(fam, 0) + 1
    print(f"\nRegenerated {len(written)} plots -> {out_dir}")
    for fam, n in sorted(families.items()):
        print(f"  {fam:24s} {n}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m adept.warpx.regen",
        description="Regenerate the WarpX canned plot set from saved NetCDF artifacts.",
    )
    ap.add_argument("src", help="run dir (containing binary/ or diags/) or a binary/ NetCDF dir")
    ap.add_argument("-o", "--out", default=None, help="output dir (default <run>/plots_regen)")
    ap.add_argument("--no-config", action="store_true", help="ignore the run's config.yaml output block")
    ap.add_argument(
        "--v-th", type=float, default=None, help="electron thermal velocity for the Langmuir overlay on omega-k plots"
    )
    ap.add_argument(
        "--omega-k-zoom",
        type=float,
        default=None,
        help="(k, omega) half-width [omega_p] for the equal-aspect lower omega-k panel",
    )
    ap.add_argument(
        "--no-zoom",
        action="store_true",
        help="use the full Nyquist window for the lower omega-k panel (omega_k_zoom=None)",
    )
    ap.add_argument("--dpi", type=int, default=None, help="figure DPI")
    ap.add_argument("--n-panels", type=int, default=None, help="panels for faceted plots")
    args = ap.parse_args(argv)

    overrides: dict = {}
    if args.v_th is not None:
        overrides["v_th"] = args.v_th
    if args.dpi is not None:
        overrides["dpi"] = args.dpi
    if args.n_panels is not None:
        overrides["n_panels"] = args.n_panels
    if args.no_zoom:
        overrides["omega_k_zoom"] = None  # explicit disable
    elif args.omega_k_zoom is not None:
        overrides["omega_k_zoom"] = args.omega_k_zoom

    out_dir = Path(args.out) if args.out else default_out_dir(args.src)
    written = regenerate(args.src, out_dir=out_dir, use_config=not args.no_config, **overrides)
    _summarize(written, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
