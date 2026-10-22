"""``BaseWarpX`` — the ADEPT module that drives WarpX."""

from __future__ import annotations

import math
from typing import Any

from adept._base_ import ADEPTModule
from adept.normalization import UREG, skin_depth_normalization
from adept.warpx import deck as _deck
from adept.warpx import post as _post
from adept.warpx import runner as _runner


class BaseWarpX(ADEPTModule):
    """Wraps an external WarpX binary as an adept solver.

    The WarpX native inputs file is the canonical simulation spec. The YAML
    manifest layered on top supplies MLflow metadata, the binary path, MPI
    rank count, the physical reference density that fixes the normalization,
    and optional in-place deck overrides.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        warpx_cfg = cfg.get("warpx", {})
        deck_path = warpx_cfg.get("deck")
        if not deck_path:
            raise ValueError("BaseWarpX: cfg['warpx']['deck'] is required")

        self._deck = _deck.parse_deck_file(deck_path)
        overrides = warpx_cfg.pop("overrides", None) or {}
        if overrides:
            _deck.merge_overrides(self._deck, overrides)

        # Surface the parsed (post-override) deck inside cfg so adept's
        # log_params picks every parameter up as a flat MLflow param. The raw
        # overrides dict is popped above; the applied values now live verbatim
        # under cfg["deck"].
        cfg["deck"] = _deck.deck_to_flat_dict(self._deck)

    def write_units(self) -> dict:
        """Derive normalized reference scales from SI inputs.

        WarpX decks are SI throughout and carry no reference density, so the
        normalization that makes runs comparable with the other adept solvers
        (time in ``1/wp0``, length in the skin depth ``c/wp0``, velocity in
        ``c``, momenta in ``m_e c``) comes from the *manifest*:
        ``warpx.reference_density``, either a pint-parsable string
        (``"9.05e21 /cc"``) or a bare number interpreted as cm^-3 (matching
        the OSIRIS ``simulation.n0`` convention). When absent, the deck's
        ``my_constants.n0`` — SI, m^-3, the constant SRS decks use inside
        ``density_function`` expressions — is used as a fallback.

        When the deck launches a laser (``lasers.names`` with ``wavelength``
        and ``e_max``), the physical drive scales are added too: ``w_laser``
        (rad/s), ``laser_wavelength`` (nm), ``laser_a0`` and
        ``laser_intensity`` — the peak intensity of a linearly polarized
        drive in W/cm^2 (``I = eps0 c e_max^2 / 2``, equivalently the ICF
        convention ``I * lambda_um^2 = 1.37e18 * a0^2``).
        """
        warpx_cfg = self.cfg.get("warpx", {})
        n0 = warpx_cfg.get("reference_density")
        if n0 is not None:
            # A unitless value — a number, or a string YAML kept as a string
            # ("9.05e21" is not a YAML float without the e+ sign) — is cm^-3.
            n0_q = UREG.Quantity(n0) if isinstance(n0, str) else UREG.Quantity(float(n0), "1/cc")
            if n0_q.dimensionless:
                n0_q = n0_q.magnitude * UREG.cc**-1
        elif self._deck.get("my_constants.n0") is not None:
            n0_q = UREG.Quantity(f"{self._deck['my_constants.n0']} / m^3")
        else:
            return {}
        norm = skin_depth_normalization(n0_q)

        quants: dict[str, Any] = {
            "wp0": (1 / norm.tau).to("rad/s"),
            "tp0": norm.tau.to("fs"),
            "n0": norm.n0.to("1/cc"),
            "v0": norm.v0.to("m/s"),
            "x0": norm.L0.to("nm"),
            "c_light": norm.speed_of_light_norm(),  # == 1.0; skin-depth norm uses v0 = c
            "beta": 1.0 / norm.speed_of_light_norm(),
        }

        lo = self._deck_array("geometry.prob_lo")
        hi = self._deck_array("geometry.prob_hi")
        if lo is not None and hi is not None:
            quants["box_length"] = ((float(hi[0]) - float(lo[0])) * UREG.m).to("micron")

        stop_time = self._deck.get("stop_time")
        if stop_time is not None:
            quants["sim_duration"] = (float(stop_time) * UREG.s).to("ps")

        # Laser drive in physical / ICF units, from the first laser's SI
        # wavelength and peak field. a0 = e E / (m_e c w_laser).
        laser = self._first_laser_name()
        if laser is not None:
            wavelength = self._deck.get(f"{laser}.wavelength")
            e_max = self._deck.get(f"{laser}.e_max")
            a0 = self._deck.get(f"{laser}.a0")  # WarpX requires exactly one of e_max / a0
            if wavelength is not None:
                w_laser = 2.0 * math.pi * UREG.c / (float(wavelength) * UREG.m)
                quants["w_laser"] = w_laser.to("rad/s")
                quants["laser_wavelength"] = (float(wavelength) * UREG.m).to("nm")
                e_peak = None
                if e_max is not None:
                    e_peak = float(e_max) * UREG.V / UREG.m
                elif a0 is not None:
                    e_peak = float(a0) * UREG.m_e * UREG.c * w_laser / UREG.e
                if e_peak is not None:
                    quants["laser_a0"] = (UREG.e * e_peak / (UREG.m_e * UREG.c * w_laser)).to("dimensionless").magnitude
                    quants["laser_intensity"] = (e_peak**2 * UREG.epsilon_0 * UREG.c / 2.0).to("W/cm^2")

        self.cfg.setdefault("units", {})["derived"] = quants
        return quants

    def get_derived_quantities(self) -> dict:
        """Lift a few useful scalars out of the deck for MLflow visibility.

        WarpX computes ``dt`` internally from the CFL number, so the step
        count for a ``stop_time``-terminated run is an estimate based on the
        FDTD Yee limit ``dt = cfl / (c sqrt(sum 1/dx_d^2))`` — the 1D
        ``cfl * dx / c`` generalized to the run's dimensionality (the 1D
        relation applied to a 2D run under-counted steps by sqrt(2)).
        """
        derived: dict[str, Any] = {}
        nx = self._deck_array("amr.n_cell")
        lo = self._deck_array("geometry.prob_lo")
        hi = self._deck_array("geometry.prob_hi")
        cfl = self._deck.get("warpx.cfl")
        max_step = self._deck.get("max_step")
        stop_time = self._deck.get("stop_time")

        if nx and lo is not None and hi is not None:
            derived["dx"] = [(float(hi[d]) - float(lo[d])) / int(nx[d]) for d in range(len(nx))]
            if cfl is not None:
                c_si = 299792458.0
                inv_sq = sum(1.0 / float(d) ** 2 for d in derived["dx"] if float(d) > 0)
                derived["dt_est"] = float(cfl) / (c_si * inv_sq**0.5)
                if stop_time is not None:
                    derived["num_steps_est"] = int(float(stop_time) / derived["dt_est"])
        if max_step is not None:
            derived["num_steps"] = int(max_step)

        # Grid resolution in skin depths when the normalization is fixed.
        x0 = ((self.cfg.get("units") or {}).get("derived") or {}).get("x0")
        if "dx" in derived and x0 is not None:
            x0_m = x0.to("m").magnitude
            derived["dx_over_skin_depth"] = [d / x0_m for d in derived["dx"]]

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
        warpx_cfg = self.cfg.get("warpx", {})
        binary = _runner.discover_binary(
            warpx_cfg.get("binary"),
            dim=self._infer_dim(),
        )
        result = _runner.run_warpx(
            _deck.render_deck(self._deck),
            binary=binary,
            mpi_ranks=int(warpx_cfg.get("mpi_ranks", 1)),
            run_root=warpx_cfg.get("run_root", "./checkpoints"),
            launcher=warpx_cfg.get("mpi_launcher", "srun"),
            extra_mpi_args=warpx_cfg.get("extra_mpi_args"),
        )
        return {"solver result": result}

    def post_process(self, run_output: dict, td: str) -> dict:
        return _post.collect(run_output, self.cfg, td)

    def vg(self, trainable_modules: dict, args: dict):
        raise NotImplementedError("WarpX is not differentiable inside adept")

    # --- helpers ----------------------------------------------------------

    def _deck_array(self, key: str) -> list | None:
        """Look up a deck value and return it as a list (scalars wrapped)."""
        v = self._deck.get(key)
        if v is None:
            return None
        return v if isinstance(v, list) else [v]

    def _first_laser_name(self) -> str | None:
        names = self._deck_array("lasers.names")
        if names:
            return str(names[0])
        return None

    def _infer_dim(self) -> int | None:
        """Read dimensionality from ``geometry.dims``, else count ``amr.n_cell``."""
        dims = self._deck.get("geometry.dims")
        if dims is not None:
            try:
                return int(dims)
            except (TypeError, ValueError):
                return None  # e.g. "RZ"
        nx = self._deck_array("amr.n_cell")
        return len(nx) if nx else None
