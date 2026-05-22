"""``BaseOsiris`` — the ADEPT module that drives OSIRIS."""

from __future__ import annotations

from typing import Any

from adept._base_ import ADEPTModule
from adept.osiris import deck as _deck
from adept.osiris import post as _post
from adept.osiris import runner as _runner


class BaseOsiris(ADEPTModule):
    """Wraps an external OSIRIS binary as an adept solver.

    The OSIRIS native deck is the canonical simulation spec. The YAML
    manifest layered on top supplies MLflow metadata, the binary path,
    MPI rank count, and optional in-place deck overrides.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        osiris_cfg = cfg.get("osiris", {})
        deck_path = osiris_cfg.get("deck")
        if not deck_path:
            raise ValueError(
                "BaseOsiris: cfg['osiris']['deck'] is required"
            )

        self._sections = _deck.parse_deck_file(deck_path)
        overrides = osiris_cfg.pop("overrides", None) or {}
        if overrides:
            _deck.merge_overrides(self._sections, overrides)

        # Surface the parsed (post-override) deck inside cfg so adept's
        # log_params picks every parameter up as a flat MLflow param.
        # The raw overrides dict is intentionally popped above to avoid
        # confusing log_params/flatdict with integer keys; the applied
        # values now live verbatim under cfg["deck"].
        cfg["deck"] = _deck.deck_to_flat_dict(self._sections)

    def write_units(self) -> dict:
        return {}

    def get_derived_quantities(self) -> dict:
        """Lift a few useful scalars out of the deck for MLflow visibility."""
        grid = dict(self._iter_first_section("grid"))
        time = dict(self._iter_first_section("time"))
        time_step = dict(self._iter_first_section("time_step"))
        space = dict(self._iter_first_section("space"))

        derived: dict[str, Any] = {}
        nx = self._first_array_value(grid, "nx_p")
        xmin = self._first_array_value(space, "xmin")
        xmax = self._first_array_value(space, "xmax")
        dt = time_step.get("dt")
        tmax = time.get("tmax")

        if nx and xmin is not None and xmax is not None:
            derived["dx"] = [
                (xmax[d] - xmin[d]) / nx[d] for d in range(len(nx))
            ]
            if dt is not None and nx:
                derived["cfl_ratio"] = float(dt) / min(derived["dx"])
        if dt is not None and tmax is not None:
            try:
                derived["num_steps"] = int(float(tmax) / float(dt))
            except (TypeError, ValueError):
                pass
        if derived:
            self.cfg.setdefault("derived", {}).update(derived)
        return derived

    def get_solver_quantities(self) -> None:
        return None

    def init_state_and_args(self) -> dict:
        return {}

    def init_diffeqsolve(self) -> None:
        return None

    def init_modules(self) -> dict:
        return {}

    def __call__(self, trainable_modules: dict, args: dict) -> dict:
        osiris_cfg = self.cfg.get("osiris", {})
        binary = _runner.discover_binary(
            osiris_cfg.get("binary"),
            dim=self._infer_dim(),
        )
        mpi_ranks = int(osiris_cfg.get("mpi_ranks", 1))
        run_root = osiris_cfg.get("run_root", "./osiris_runs")
        deck_text = _deck.render_deck(self._sections)
        result = _runner.run_osiris(
            deck_text,
            binary=binary,
            mpi_ranks=mpi_ranks,
            run_root=run_root,
            extra_mpi_args=osiris_cfg.get("extra_mpi_args"),
        )
        return {"solver result": result}

    def post_process(self, run_output: dict, td: str) -> dict:
        return _post.collect(run_output, self.cfg, td)

    def vg(self, trainable_modules: dict, args: dict):
        raise NotImplementedError(
            "OSIRIS is not differentiable inside adept"
        )

    # --- helpers ----------------------------------------------------------

    def _iter_first_section(self, name: str) -> dict:
        for sec_name, params in self._sections:
            if sec_name == name:
                return params
        return {}

    @staticmethod
    def _first_array_value(params: dict, base_key: str) -> list | None:
        """Look up either ``base_key`` or ``base_key(...)`` and return as list."""
        for k, v in params.items():
            if k == base_key or k.startswith(base_key + "("):
                if isinstance(v, list):
                    return v
                return [v]
        return None

    def _infer_dim(self) -> int | None:
        """Best-effort: read dimensionality from ``grid.nx_p(1:D)``."""
        grid = self._iter_first_section("grid")
        for k in grid:
            if k.startswith("nx_p"):
                v = grid[k]
                return len(v) if isinstance(v, list) else 1
        return None
