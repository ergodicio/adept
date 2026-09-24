import numpy as np
from jax import Array, lax
from jax import numpy as jnp

from adept._base_ import get_envelope
from adept._lpse2d.core import epw, laser
from adept._lpse2d.core.epw import LEDGER_CHANNELS, LEDGER_KEY
from adept._lpse2d.core.light import CoupledLight
from adept._lpse2d.core.raman import RamanLight
from adept._lpse2d.core.timeline import Linear


class SplitStep:
    """
    This class contains the function that updates the state

    All the pushers are chosen and initialized here and a single time-step is defined here.

    :param cfg:
    :return:
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dt = cfg["grid"]["dt"]
        self.wp0 = cfg["units"]["derived"]["wp0"]
        self.epw = epw.SpectralEPWSolver(cfg)
        self.light = laser.Light(cfg)
        # terms.epw.solver: "separate" (the four envelope equations, MATLAB / LPSE spectral)
        # or "combined" (LPSE lw.solver = combined: one wp0-enveloped field for Raman light
        # and EPW, required by LPSE whenever TPD and SRS are both on)
        self.epw_solver = str(cfg["terms"]["epw"].get("solver", "separate"))
        if self.epw_solver == "combined":
            from adept._lpse2d.core.combined import CombinedSolver

            self.combined = CombinedSolver(cfg)
        elif self.epw_solver != "separate":
            raise ValueError(f"terms.epw.solver must be 'separate' or 'combined', got {self.epw_solver!r}")
        # the Raman scattered light is evolved iff the SRS source term is on; with
        # terms.light.pump_depletion the pump is evolved too (one coupled solver)
        self.pump_depletion = cfg["terms"].get("light", {}).get("pump_depletion", False)
        srs_on = cfg["terms"]["epw"]["source"].get("srs", False)
        # terms.light.solver: the MATLAB staggered finite-difference scheme (fd) or the
        # LPSE spectral propagator (spectral, spectral_light.py)
        self.light_solver = str(cfg["terms"].get("light", {}).get("solver", "fd"))
        if self.light_solver == "spectral":
            from adept._lpse2d.core.spectral_light import SpectralCoupledLight, SpectralRamanLight

            coupled_cls, raman_cls = SpectralCoupledLight, SpectralRamanLight
        elif self.light_solver == "fd":
            coupled_cls, raman_cls = CoupledLight, RamanLight
        else:
            raise ValueError(f"terms.light.solver must be 'fd' or 'spectral', got {self.light_solver!r}")
        if self.pump_depletion:
            self.coupled_light = coupled_cls(cfg)
            self.raman = None
        else:
            self.raman = raman_cls(cfg) if srs_on else None
        if cfg["terms"].get("qle", {}).get("active", False):
            from adept._lpse2d.core.qle import QuasilinearEvolution

            if cfg["terms"].get("hpe", {}).get("active", False):
                raise ValueError("terms.qle and terms.hpe both evolve the Landau rate: enable one of them")
            self.qle = QuasilinearEvolution(cfg)
        else:
            self.qle = None
        if cfg["terms"].get("hpe", {}).get("active", False):
            from adept._lpse2d.core.hpe import HybridParticleEvolution

            self.hpe = HybridParticleEvolution(cfg)
        else:
            self.hpe = None
        if cfg["terms"].get("iaw", {}).get("active", False):
            from adept._lpse2d.core.iaw import IonAcousticWave

            self.iaw = IonAcousticWave(cfg)
        else:
            self.iaw = None
        # LPSE interpolateSourcesInTime (default on): the light reads the EPW potential and the ion
        # density, the EPW the ion density, linearly interpolated in time (core/timeline.py)
        self.interpolate_light = bool(cfg["terms"].get("light", {}).get("interpolate_sources", True))
        self.interpolate_epw = bool(cfg["terms"]["epw"].get("interpolate_sources", True))
        # HPE particle/histogram keys are real and stay out of this list
        self.complex_state_vars = ["E0", "epw", "E1"]
        # terms.epw.energy_ledger: accumulate the per-operation EPW energy changes in the state
        self.energy_ledger = bool(cfg["terms"]["epw"].get("energy_ledger", False))
        self.boundary_envelope = cfg["grid"]["absorbing_boundaries"]
        self.one_over_ksq = cfg["grid"]["one_over_ksq"]
        self.zero_mask = cfg["grid"]["zero_mask"]
        self.low_pass_filter = cfg["grid"]["low_pass_filter"]
        self.k_sq = cfg["grid"]["kx"][:, None] ** 2 + cfg["grid"]["ky"][None, :] ** 2
        self.one_over_ksq = cfg["grid"]["one_over_ksq"]

        self.envelope_density = cfg["units"]["envelope density"]
        self.e = cfg["units"]["derived"]["e"]
        self.me = cfg["units"]["derived"]["me"]
        self.w0 = cfg["units"]["derived"]["w0"]
        self.phi_laplacian = "spectral"  # hard coded for now, can be implemented to config if ever necessary
        self.background_density = cfg["grid"]["background_density"]

    def _unpack_y_(self, y: dict[str, Array]) -> dict[str, Array]:
        new_y = {}
        for k in y.keys():
            if k in self.complex_state_vars:
                new_y[k] = y[k].view(jnp.complex128)
            else:
                new_y[k] = y[k].view(jnp.float64)
        return new_y

    def _pack_y_(self, y: dict[str, Array], new_y: dict[str, Array]) -> tuple[dict[str, Array], dict[str, Array]]:
        for k in y.keys():
            y[k] = y[k].view(jnp.float64)
            new_y[k] = new_y[k].view(jnp.float64)

        return y, new_y

    def get_envelope_coefficient(self, envelope_args, t):
        return get_envelope(
            envelope_args["tr"],
            envelope_args["tr"],
            envelope_args["tc"] - envelope_args["tw"] / 2,
            envelope_args["tc"] + envelope_args["tw"] / 2,
            t,
        )

    def iaw_step(self, y, t):
        """The IAW step at ``t``, applied every ``stride``-th EPW step (LPSE strides the IAW
        solver, advancing by stride*dt) and inside ``[t_start, t_stop)`` (LPSE
        ``iaw.startEvolvingTime`` / ``stopEvolvingTime``); the direct call when nothing gates it."""
        gates = []
        if self.iaw.stride > 1:
            gates.append(jnp.round(t / self.dt).astype(int) % self.iaw.stride == 0)
        if self.iaw.t_start is not None:
            gates.append(t >= self.iaw.t_start)
        if self.iaw.t_stop is not None:
            gates.append(t < self.iaw.t_stop)
        if not gates:
            return self.iaw(y, t)
        active = gates[0]
        for gate in gates[1:]:
            active = active & gate
        return lax.cond(active, lambda yy: self.iaw(yy, t), lambda yy: yy, y)

    def iaw_first(self, y, t, drive=None):
        """The IAW step at the start of the EPW step, driven by the fields at ``t`` (LPSE
        ``ZakharovSolver::evolve`` advances the IAW, then the Langmuir waves, then the light).
        ``drive`` is the state the ponderomotive drive is formed from (default ``y``). Returns the
        state with the new IAW entries and the ends ``(old, new)`` of the IAW step with the
        fractions of it this EPW step spans: the IAW advances every ``stride`` EPW steps, and in
        between the waves read the density between ``Nelf_old`` and ``Nelf`` (``linearInterp``)."""
        n = self.iaw.stride
        before = y["iaw_density"]
        out = self.iaw_step(y if drive is None else drive, t)
        y = {**y, **{k: v for k, v in out.items() if k.startswith("iaw_")}}
        if n == 1:
            return y, (before, y["iaw_density"], 0.0, 1.0)
        m = jnp.round(t / self.dt).astype(int) % n
        old = jnp.where(m == 0, before, y["iaw_density_old"])
        y["iaw_density_old"] = old
        return y, (old, y["iaw_density"], m / n, (m + 1) / n)

    def light_split_step(self, t, y, driver_args, phi_k=None, iaw_density=None):
        """The light over the EPW step. ``phi_k`` / ``iaw_density``: the EPW potential and ion
        density as arrays or ``timeline.Linear`` (default: the state's, held fixed)."""
        phi_k = y["epw"] if phi_k is None else phi_k
        iaw_density = y.get("iaw_density") if iaw_density is None else iaw_density
        if self.pump_depletion:
            # the pump is a dynamic field sourced by its boundary injector; both light
            # waves advance inside one staggered update (absorbers applied per sub-step)
            y["E0"], y["E1"] = self.coupled_light(
                t,
                y["E0"],
                y["E1"],
                phi_k,
                driver_args["E0"],
                driver_args.get("E1"),
                iaw_density,
            )
            return y

        if "E0" in driver_args:

            def E0_fn(this_t):
                t_coeff = self.get_envelope_coefficient(driver_args["E0"], this_t)
                return t_coeff * self.light.laser_update(this_t, y, driver_args["E0"])

            y["E0"] = E0_fn(t)
        else:
            E0_now = y["E0"]

            def E0_fn(this_t):
                return E0_now

        if self.raman is not None:
            # evolve the Raman light; absorbing boundaries are applied per light sub-step inside
            y["E1"] = self.raman(t, y["E1"], E0_fn, phi_k, driver_args.get("E1"), iaw_density)
        else:
            y["E1"] *= self.boundary_envelope[..., None]

        return y

    def combined_step(self, t, y, driver_args, iaw_density=None):
        """One EPW step of the combined solver: pump (prescribed or evolved), the combined
        Raman + EPW field, and the derived potential. ``iaw_density``: an array or
        ``timeline.Linear`` (default: the state's)."""
        if self.pump_depletion:
            E0_fn = None
        elif "E0" in driver_args:

            def E0_fn(this_t):
                t_coeff = self.get_envelope_coefficient(driver_args["E0"], this_t)
                return t_coeff * self.light.laser_update(this_t, y, driver_args["E0"])

        else:
            E0_now = y["E0"]

            def E0_fn(this_t):
                return E0_now

        solver_y = y if iaw_density is None else {**y, "iaw_density": iaw_density}
        y["E0"], y["E1"], y["epw"] = self.combined(t, solver_y, driver_args, E0_fn)
        return y

    def __call__(self, t, y, args):
        """One EPW step in LPSE's order (``ZakharovSolver::evolve``): the IAW with the fields at
        ``t``, the EPW with the light at ``t`` and the ion density at the step's middle, then the
        light with the EPW potential and ion density at each sub-step's middle -- each read
        linearly between the old and new values (``interpolate_sources``; the new ones without)."""
        # unpack y into complex128
        new_y = self._unpack_y_(y)

        iaw_ends = None
        if self.iaw is not None:
            drive = None
            if self.epw_solver == "combined":
                # the IAW ponderomotive drive sees the Raman light (transverse part) and the
                # EPW (through the derived potential) separately, not the combined field
                drive = {**new_y, "E1": self.combined.transverse(new_y["E1"])}
            new_y, iaw_ends = self.iaw_first(new_y, t, drive)
        light_iaw = None if iaw_ends is None else Linear(*iaw_ends, interpolate=self.interpolate_light)

        if self.epw_solver == "combined":
            new_y = self.combined_step(t, new_y, args["drivers"], light_iaw)
            if self.hpe is not None:
                new_y = self.hpe(t, new_y)
            if self.qle is not None:
                new_y = self.qle(t, new_y)  # y["epw"] is the potential derived from the combined field
            y, new_y = self._pack_y_(y, new_y)
            return new_y

        driver_delta = 0.0
        if "E2" in args["drivers"]:
            w_before = self.epw.energy(new_y["epw"])
            new_y["epw"] += jnp.fft.fft2(self.dt * self.epw.driver(args["drivers"]["E2"], t))
            driver_delta = self.epw.energy(new_y["epw"]) - w_before
        # epw split step with the light at t (the per-operation energy deltas for the ledger)
        phi_old = new_y["epw"]
        epw_in = new_y
        if iaw_ends is not None:
            epw_in = {**new_y, "iaw_density": Linear(*iaw_ends, interpolate=self.interpolate_epw).at(0.5)}
        new_y["epw"], deltas = self.epw.advance(t, epw_in, args)
        if self.energy_ledger:
            deltas = deltas.at[LEDGER_CHANNELS.index("driver")].set(driver_delta)
            new_y[LEDGER_KEY] = new_y[LEDGER_KEY] + deltas

        # light split step with the EPW potential between its values at t and t + dt
        phi = Linear(phi_old, new_y["epw"], interpolate=self.interpolate_light)
        new_y = self.light_split_step(t, new_y, args["drivers"], phi, light_iaw)

        # particle push + Landau-damping feedback; the gamma_L written here is the
        # rate the EPW update applies on the next step (one-step lag)
        if self.hpe is not None:
            new_y = self.hpe(t, new_y)
        # quasilinear VDF update and its Landau rate (same one-step lag; plan 2 K.1)
        if self.qle is not None:
            new_y = self.qle(t, new_y)

        # pack y into float64
        y, new_y = self._pack_y_(y, new_y)

        return new_y
