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
- **Transverse fields** ``(E_y, B_z)``: Yee FDTD (later increment).
- **Longitudinal field** ``E_x`` / charge conservation: Ampère + Esirkepov
  current deposition (later increment).
- **Shape functions / gather / deposit**: reused from
  :mod:`adept._pic1d.solvers.pushers.shape`.

Build status: the Higuera–Cary pusher and its single-particle analytic
validation tests have landed (increment 1). The Yee field solver, Esirkepov
current deposition, and the ``ergoExo`` run path follow in later increments.
"""
