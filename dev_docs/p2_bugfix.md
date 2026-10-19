# P2 bugfix handoff — blank p1p2 plots + truncated p1x1 heatmap (WarpX 2D histogram converter)

Diagnosed 2026-08-24 on the `warpx-srs-2d-follett` run
`4dd5bb478f1a417aba93968b99355703` (exp 188989, the 08-18 follett-2D run).
Both bugs live in the ParticleHistogram2D → OSIRIS-PHA conversion path in
`adept/adept/warpx/io.py` (`pic-wrapper` branch). The 1D campaigns are
unaffected (in 1D the abscissa really is z and the box is z-only), which is
why neither surfaced before the first 2D run.

## Symptoms (all verified against the MLflow artifacts)

1. **Blank p1p2 plots.** `plots/phasespace/electrons/p1p2.png` is an empty
   panel whose x-axis is labeled `x1` and spans ~430000–470000 $c/\omega_p$;
   `plots/distribution_lineouts/electrons/p1p2.png` shows flat-zero curves.
   (`plots/phasespace_evolution/electrons/p1p2.png` will be the same — not
   inspected.)
2. **Truncated p1x1 heatmap.** `plots/phasespace/electrons/p1x1.png` renders
   real data but only for $x_1 \in [0, 53.7]\ c/\omega_p$ — the first ~9.4 %
   (3 µm) of the 573.3 $c/\omega_p$ (32 µm) propagation box. All other p1x1
   plots (evolution, distribution_lineouts) are fine.

The underlying histogram *data* is fine in both cases — this is purely a
conversion/plotting-metadata problem.

## Root cause 1 — abscissa hard-coded as spatial (`x1`, meters)

`load_particle_histogram2d` (`adept/adept/warpx/io.py`, returns at
`dims=("t", "x1", ord_dim)`) always names the histogram **abscissa** `x1` and
converts its bin edges meters → code units
(`x_axis = _edge_style_axis(abs_lo, abs_hi, n_abs) / units.x0`). Only the
**ordinate** goes through `_ordinate_info` (`uz→p1`, `ux→p2`, `uy→p3`,
`log10(...)→gamma`).

- `p1x1` (deck: `histogram_function_abs = z`) → correct.
- `p1p2` (deck: `histogram_function_abs = uz`, `_ord = ux`) → the
  $u_z \in [-5, 5]\ m_e c$ bins are divided by $x_0 = 5.586\times10^{-8}$ m,
  producing a fake `x1` coordinate spanning $\pm 8.95\times10^7\ c/\omega_p$.

Verified in the uploaded `binary/PHA/p1p2/electrons.nc`: dims
`(t: 81, x1: 200, p2: 100)`, `x1 ∈ [-8.951e7, 8.951e7]`, attrs
`warpx_function_abs: 'uz'`, last dump has 4706 nonzero (negative,
charge-signed) bins — data intact.

Downstream, `_crop_spatial_to_box` (`adept/adept/osiris/plots.py:956`) trims
every `x*` dim to `[sim.XMIN[0], sim.XMAX[0]] = [0, 53.74]`. On a grid with
~$9\times10^5$-wide bins exactly one bin overlaps that interval (the
`right = max(left + 1, ...)` guarantee keeps it), leaving a single
near-invisible column centered at ~4.5e5 — hence the blank heatmap and its
weird 430000–470000 axis. The same one-column crop feeds the
distribution-lineout averaging (values ~1e-9, negative) → flat-zero panels.
Note the crop helper's docstring assumes momentum–momentum spaces pass
through untouched ("e.g. p1p2") — true for OSIRIS-native dims (`p1`, `p2`),
defeated by the converter's mislabeled `x1` dim.

## Root cause 2 — `sim.XMIN/XMAX` in WarpX (x, z) order, not x1-first

`_box_attrs` (`adept/adept/warpx/io.py:836` area) copies
`geometry.prob_lo/hi` in deck order. In 2D that is WarpX `(x, z)`, so the
`.nc` attrs read `sim.XMAX = [53.74, 573.26]` — the **transverse** extent
lands in slot 0, which the crop applies to the z (x1) axis. Commit `36a22ae`
(08-20, "2D OSIRIS-contract parity in the openPMD converter") fixed exactly
this ordering for the **field** converter but did not touch `_box_attrs` in
the histogram path. `sim.NDIMS` is also hard-coded to `1` in the histogram
attrs.

This is what truncates the (otherwise healthy) p1x1 phase-space heatmap to
the first 53.7 $c/\omega_p$, and it also supplied the [0, 53.74] crop window
that blanked p1p2 (though p1p2 would be broken by root cause 1 regardless).

## Fix plan

In `load_particle_histogram2d`:

1. Map the abscissa function through the same classifier as the ordinate:
   - spatial (`z`, and in 2D potentially `x`/`y`) → `x1`/`x2`/`x3` via the
     cyclic (z, x, y) → (1, 2, 3) relabeling, with the meters → $c/\omega_p$
     conversion and the existing $\Delta x_{bin} \cdot n_0 \cdot \Delta_{ord}$
     normalization;
   - momentum (`uz`/`ux`/`uy`) → `p1`/`p2`/`p3`, left in $m_e c$ (no
     `units.x0` division). Reuse/generalize `_ordinate_info` rather than
     duplicating it.
   Check the normalization denominator when the abscissa is momentum:
   `d_abs` is then in $m_e c$, not meters, so the "count → phase-space
   density" scale needs the same units-awareness.
2. Order `_box_attrs` output x1-first in 2D/3D (mirror what `36a22ae` did for
   fields) and set `sim.NDIMS` from the actual geometry.
3. Regression tests alongside `tests/test_warpx/test_io_2d.py`: a synthetic
   p1p2-style (uz, ux) histogram must come out as dims `(t, p1, p2)` with
   momentum coords, and survive `plots._crop_spatial_to_box` unchanged; a
   p1x1-style one in 2D must carry `sim.XMAX` x1-first.

## Validation / re-render

- Raw openPMD inputs are still on Perlmutter scratch:
  `$PSCRATCH/warpx-lpi/checkpoints/srs-2d-follett-3um/20260818T110509_305720e2/diags/reducedfiles/{p1x1,p1p2,x1log_gamma_q1,log_gamma}/`
  (source_dir attr in the .nc files). Re-convert with the fixed code and
  re-render the phasespace / phasespace_evolution / distribution_lineouts
  sets; p1p2 should show a thermal blob ($u_{th} \approx 0.044$) and p1x1
  should span the full $x_1 \in [0, 573.3]$.
- The already-uploaded `binary/PHA/p1p2/electrons.nc` has the bad
  coordinates baked in, but its values are intact — recoverable by renaming
  dims/coords if scratch ever goes away.
- Local copies of everything inspected during diagnosis (pngs + both .nc
  files) are in the bg-job tmp dir `~/.claude/jobs/f6fa03a2/tmp/` —
  ephemeral, re-download from MLflow run `4dd5bb478f1a417aba93968b99355703`
  if needed.

## Context

- The 08-20 P2 recheck (`srs-campaign/sims/warpx/srs-2d-follett-3um/NOTES.md`
  run log, `recheck_2d.py`) re-converted fields with the `36a22ae` fixes but
  the histogram (PHA) conversion path was not part of that fix.
- Related open items from the same campaign (separate from this fix): the
  FieldPoyntingFlux / s1-scalar estimators are unreliable on this run, and
  `BoundaryScraping` should be dropped from both the 2D and 1D decks (dead
  under thermal walls, 11 % of wall) — see NOTES.md 08-20 entries.

## Status — FIXED 2026-08-24

Both root causes fixed in `adept/warpx/io.py` (uncommitted on
`pic-wrapper`): the abscissa now goes through the generalized
`_ordinate_info` classifier (`uz/ux/uy → p1/p2/p3` in m_e c, no `x0`
division; `z/x/y → x1/x2/x3` spatial; unrecognized/absent falls back to
the legacy spatial-x1 reading), and `_box_attrs` orders `sim.XMIN/XMAX`
x1-first with `sim.NDIMS` from the geometry. Regression tests:
`tests/test_warpx/test_pha_2d_axes.py` (4). Validated against this run's
raw scratch data (p1p2 `(t, p1, p2)`, 4706 nonzero bins; p1x1 full
573.26 x1 extent); corrected phasespace/phasespace_evolution PNGs
uploaded over the broken artifacts in run
`4dd5bb478f1a417aba93968b99355703`. See the campaign NOTES.md 08-24
entry.
