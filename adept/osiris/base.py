"""``BaseOsiris`` — the ADEPT module that drives OSIRIS."""

from __future__ import annotations

from typing import Any

import math

from adept._base_ import ADEPTModule
from adept.normalization import UREG, skin_depth_normalization, skin_depth_normalization_from_frequency
from adept.osiris import deck as _deck
from adept.osiris import density as _density
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
            raise ValueError("BaseOsiris: cfg['osiris']['deck'] is required")

        self._sections = _deck.parse_deck_file(deck_path)
        overrides = osiris_cfg.pop("overrides", None) or {}
        if overrides:
            _deck.merge_overrides(self._sections, overrides)

        # Optional: scale the box from a target density gradient scale length
        # (mirrors adept's _lpse2d / kinetic_srs grid sizing). Runs *after*
        # overrides, so an explicit space.xmax override is superseded by the
        # gradient-derived box. The computed quantities are stashed back under
        # osiris.density.derived for MLflow provenance.
        density_cfg = osiris_cfg.get("density")
        computed = _density.apply_gradient_scale_length(self._sections, density_cfg)
        if computed is not None:
            density_cfg.setdefault("derived", {}).update(computed)

        # Surface the parsed (post-override) deck inside cfg so adept's
        # log_params picks every parameter up as a flat MLflow param.
        # The raw overrides dict is intentionally popped above to avoid
        # confusing log_params/flatdict with integer keys; the applied
        # values now live verbatim under cfg["deck"].
        cfg["deck"] = _deck.deck_to_flat_dict(self._sections)

    def write_units(self) -> dict:
        """Derive physical reference scales from the deck's ``simulation`` section.

        OSIRIS normalizes time to ``1/wp0``, length to the skin depth ``c/wp0``,
        and velocity to ``c``, where ``wp0`` is the reference plasma frequency.
        That reference is set in the deck's ``simulation`` section by either the
        density ``n0`` (cm^-3) or the frequency ``omega_p0`` (rad/s); when both
        are present OSIRIS uses ``n0``, so we do too. The returned dict mirrors
        the density-derived keys the other adept solvers log to ``units.yaml`` so
        OSIRIS runs are comparable in MLflow. OSIRIS has no single global
        reference temperature (species carry their own per-species thermal
        momenta), so the temperature-dependent keys (``T0``/``nuee``/
        ``logLambda_ee``) are intentionally omitted.

        When the deck launches a laser (an ``antenna`` / ``zpulse_speckle`` /
        ``zpulse`` section with ``a0`` and ``omega0``), the physical drive
        scales are added too: ``w_laser`` (rad/s), ``laser_wavelength`` (nm),
        ``laser_a0`` and ``laser_intensity`` — the peak intensity of a
        linearly polarized drive in W/cm^2 (the ICF convention
        ``I * lambda_um^2 = 1.37e18 * a0^2``).
        """
        sim = self._iter_first_section("simulation")
        n0 = sim.get("n0")
        omega_p0 = sim.get("omega_p0")
        if n0 is not None:
            norm = skin_depth_normalization(f"{n0} / cc")
        elif omega_p0 is not None:
            norm = skin_depth_normalization_from_frequency(f"{omega_p0} rad/s")
        else:
            return {}

        quants: dict[str, Any] = {
            "wp0": (1 / norm.tau).to("rad/s"),
            "tp0": norm.tau.to("fs"),
            "n0": norm.n0.to("1/cc"),
            "v0": norm.v0.to("m/s"),
            "x0": norm.L0.to("nm"),
            "c_light": norm.speed_of_light_norm(),  # == 1.0; OSIRIS normalizes v to c
            "beta": 1.0 / norm.speed_of_light_norm(),
        }

        space = self._iter_first_section("space")
        xmin = self._first_array_value(space, "xmin")
        xmax = self._first_array_value(space, "xmax")
        if xmin is not None and xmax is not None:
            quants["box_length"] = ((xmax[0] - xmin[0]) * norm.L0).to("micron")

        time = self._iter_first_section("time")
        tmax = time.get("tmax")
        if tmax is not None:
            tmin = time.get("tmin", 0.0)
            quants["sim_duration"] = ((float(tmax) - float(tmin)) * norm.tau).to("ps")

        # Laser drive in physical / ICF units. The deck's laser section carries
        # a0 and omega0 (in wp0 units); with wp0 fixed above these give the
        # laser frequency, the vacuum wavelength lambda = 2 pi c / w_laser, and
        # the peak intensity of a linearly polarized drive,
        # I = eps0 c E0^2 / 2 with E0 = a0 m_e c w_laser / e — equivalently the
        # usual ICF convention I[W/cm^2] * lambda[um]^2 = 1.37e18 * a0^2.
        # Same section priority as the SRS postproc: antenna (1D), then
        # zpulse_speckle / zpulse (2D).
        for sec_name in ("antenna", "zpulse_speckle", "zpulse"):
            laser = self._iter_first_section(sec_name)
            if laser.get("a0") is not None and laser.get("omega0") is not None:
                a0, omega0 = float(laser["a0"]), float(laser["omega0"])
                w_laser = omega0 / norm.tau
                e_peak = a0 * w_laser * UREG.m_e * UREG.c / UREG.e
                quants["w_laser"] = w_laser.to("rad/s")
                quants["laser_wavelength"] = (2.0 * math.pi * UREG.c / w_laser).to("nm")
                quants["laser_a0"] = a0
                quants["laser_intensity"] = (e_peak**2 * UREG.epsilon_0 * UREG.c / 2.0).to("W/cm^2")
                break

        self.cfg.setdefault("units", {})["derived"] = quants
        return quants

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
            derived["dx"] = [(xmax[d] - xmin[d]) / nx[d] for d in range(len(nx))]
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
        run_root = osiris_cfg.get("run_root", "./checkpoints")
        deck_text = _deck.render_deck(self._sections)
        result = _runner.run_osiris(
            deck_text,
            binary=binary,
            mpi_ranks=mpi_ranks,
            run_root=run_root,
            launcher=osiris_cfg.get("mpi_launcher", "srun"),
            extra_mpi_args=osiris_cfg.get("extra_mpi_args"),
            stream_convert=bool(osiris_cfg.get("stream_convert", True)),
            stream_poll_s=float(osiris_cfg.get("stream_poll_s", 10.0)),
            stage_root=osiris_cfg.get("stage_root"),
            stage_discard_h5=bool(osiris_cfg.get("stage_discard_h5", False)),
        )
        return {"solver result": result}

    def post_process(self, run_output: dict, td: str) -> dict:
        return _post.collect(run_output, self.cfg, td)

    def vg(self, trainable_modules: dict, args: dict):
        raise NotImplementedError("OSIRIS is not differentiable inside adept")

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
