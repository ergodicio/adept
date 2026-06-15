"""Offline regeneration of the OSIRIS canned plot set from saved NetCDFs.

When a run finishes, ``post.collect`` converts every diagnostic's time history
into a per-diagnostic NetCDF under ``binary/`` (mirroring the OSIRIS ``MS/``
layout) and renders the canned plot set. This module regenerates that *same*
plot set from the ``binary/`` NetCDFs alone — no rerun and no raw ``MS/`` HDF5
tree — so the plotting code in :mod:`adept.osiris.plots` can be iterated on
quickly against real data.

The heavy lifting already lives in :func:`adept.osiris.io.list_diagnostics` /
:func:`adept.osiris.io.load_series`, which transparently dispatch between an
``MS/`` HDF5 tree and a ``binary/`` NetCDF directory. Regeneration is therefore
just :func:`adept.osiris.plots.save_canned_plots` pointed at the NetCDF
directory; this module supplies the ergonomics: locating the ``binary/`` dir,
reading plot knobs from the run's ``config.yaml``, and a CLI.

Usage::

    python -m adept.osiris.regen <run-or-binary-dir> [--out DIR] [options]

Examples::

    # Regenerate into <dir>/plots_regen, reading output.* from config.yaml if present
    python -m adept.osiris.regen scratch/scan-a0-0.003

    # Overlay the Langmuir branch (electron uth) and tighten the omega-k zoom
    python -m adept.osiris.regen scratch/scan-a0-0.003 --v-th 0.0885 --omega-k-zoom 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adept.osiris import plots as _plots


def find_binary_dir(path: str | Path) -> Path:
    """Resolve the NetCDF diagnostics directory from a run dir or the dir itself.

    Accepts a run directory that contains a ``binary/`` subdir (the common
    case — the artifact layout), or a directory that *is* the NetCDF tree
    already. Returns the directory to hand to ``save_canned_plots``.
    """
    path = Path(path)
    if (path / "binary").is_dir():
        return path / "binary"
    return path


def default_out_dir(src: str | Path) -> Path:
    """Where regenerated plots land by default: ``<run-dir>/plots_regen``."""
    src = Path(src)
    run_dir = src if (src / "binary").is_dir() else src.parent
    return run_dir / "plots_regen"


def load_output_cfg(src: str | Path) -> dict:
    """Best-effort read of the manifest ``output:`` block for a run.

    Looks for ``config.yaml`` at ``src`` or its parent (so it is found whether
    ``src`` is the run dir or the ``binary/`` dir). Returns the ``output`` dict,
    or ``{}`` when no config is present / readable / has no ``output`` block.
    """
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
    ``config.yaml`` / ``derived_config.yaml``) or a ``binary/`` NetCDF
    directory. Plot knobs default to the run's ``output:`` config (unless
    ``use_config=False``); any keyword in ``overrides`` (``v_th``,
    ``omega_k_zoom``, ``dpi``, ``n_panels``) is applied on top verbatim — pass
    only the knobs you mean to set. Writes PNGs under ``out_dir`` (default
    ``<run-dir>/plots_regen``) and returns the ``{plot-name: path}`` map.

    SRS-specific plots (laser energy budget, distribution lineouts) live in
    osiris-lpi; regenerate those with ``python -m osiris_lpi.regen``.
    """
    binary = find_binary_dir(src)
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir(src)
    kwargs = _plots.canned_plot_kwargs(load_output_cfg(src) if use_config else None)
    kwargs.update(overrides)
    return _plots.save_canned_plots(binary, out_dir, **kwargs)


def _summarize(written: dict[str, Path], out_dir: Path) -> None:
    """Print a compact per-family summary of what was regenerated."""
    families: dict[str, int] = {}
    for name in written:
        fam = name.split("/", 1)[0]
        families[fam] = families.get(fam, 0) + 1
    print(f"\nRegenerated {len(written)} plots -> {out_dir}")
    for fam, n in sorted(families.items()):
        print(f"  {fam:24s} {n}")


def _cli_overrides(args: argparse.Namespace) -> dict:
    """Collect only the plot knobs the user explicitly set on the command line."""
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
    return overrides


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m adept.osiris.regen",
        description="Regenerate the OSIRIS canned plot set from saved NetCDF artifacts.",
    )
    ap.add_argument("src", help="run dir (containing binary/) or a binary/ NetCDF dir")
    ap.add_argument("-o", "--out", default=None, help="output dir (default <run>/plots_regen)")
    ap.add_argument("--no-config", action="store_true",
                    help="ignore the run's config.yaml output block")
    ap.add_argument("--v-th", type=float, default=None,
                    help="electron thermal velocity for the Langmuir overlay on omega-k plots")
    ap.add_argument("--omega-k-zoom", type=float, default=None,
                    help="(k, omega) half-width [omega_p] for the equal-aspect lower omega-k panel")
    ap.add_argument("--no-zoom", action="store_true",
                    help="use the full Nyquist window for the lower omega-k panel (omega_k_zoom=None)")
    ap.add_argument("--dpi", type=int, default=None, help="figure DPI")
    ap.add_argument("--n-panels", type=int, default=None, help="panels for faceted plots")
    args = ap.parse_args(argv)

    out_dir = Path(args.out) if args.out else default_out_dir(args.src)
    written = regenerate(
        args.src, out_dir=out_dir, use_config=not args.no_config, **_cli_overrides(args)
    )
    _summarize(written, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
