"""adept ``lpse2d`` vs original-LPSE parity tooling: reference-run location and the
run / compare / log harness. ``python -m adept._lpse2d.parity --help`` for the CLI."""

from .harness import (
    DEFAULT_OVERRIDES,
    DEFAULT_WINDOWS,
    DeckRun,
    compare,
    compare_hpe_flux,
    energy_floor_crossings,
    growth_rate,
    hpe_wall_power,
    log_reference,
    merge_overrides,
    parse_windows,
    read_flux,
    run_deck,
    translate_deck,
)
from .reference import (
    LPSE_PARITY_EXPERIMENT,
    deck_path,
    download_reference,
    download_run_artifacts,
    find_reference_run,
    lpse_root,
    reference_cache_dir,
    reference_run_dir,
)

__all__ = [
    "DEFAULT_OVERRIDES",
    "DEFAULT_WINDOWS",
    "LPSE_PARITY_EXPERIMENT",
    "DeckRun",
    "compare",
    "compare_hpe_flux",
    "deck_path",
    "download_reference",
    "download_run_artifacts",
    "energy_floor_crossings",
    "find_reference_run",
    "growth_rate",
    "hpe_wall_power",
    "log_reference",
    "lpse_root",
    "merge_overrides",
    "parse_windows",
    "read_flux",
    "reference_cache_dir",
    "reference_run_dir",
    "run_deck",
    "translate_deck",
]
