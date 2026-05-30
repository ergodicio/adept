"""Relativistic 1D2V electromagnetic PIC (``_empic1d``).

A relativistic, electromagnetic particle-in-cell solver in one spatial
dimension with two momentum components ``(p_x, p_y)`` and fields
``(E_x, E_y, B_z)`` (linear polarization). It is being built to enable
*differentiable* optimization of accelerator profiles:

- **PWFA** — optimize a drive-beam longitudinal current profile at fixed
  charge for transformer ratio (first target).
- **LWFA** — optimize a laser temporal profile at fixed pulse energy
  (second target).

Design (see ``memory/project_empic1d.md`` for the full rationale):

- **Pusher**: relativistic Higuera–Cary (volume-preserving; correct E×B
  drift). Implemented in :mod:`adept._empic1d.solvers.pushers.push`.
- **Transverse fields** ``(E_y, B_z)``: Yee FDTD, with a laser soft-source
  antenna (:mod:`adept._empic1d.laser`).
- **Longitudinal field** ``E_x`` / charge conservation: Ampère + Esirkepov
  current deposition.
- **Shape functions / gather / deposit**: reused from
  :mod:`adept._pic1d.solvers.pushers.shape`.

Build status: complete relativistic EM-PIC. The longitudinal path
(``longitudinal_step``) and the full electromagnetic path (``em_step``) are both
implemented and validated (single-particle orbits, Gauss's law, plasma
oscillation, PWFA wake vs linear theory, differentiable transformer-ratio
optimization, vacuum/plasma EM dispersion, laser propagation). Open boundaries
(Mur ABC) and the ``ergoExo``/MLflow run path are intentionally out of scope: the
solver is consumed as differentiable kernels (driven by ``jax.lax.scan``), which
is the right surface for the downstream inverse-problem studies.
"""
