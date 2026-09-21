# Discrete electric work

The nonrelativistic electric operator corrects a known second-order radial
truncation error in its isotropic energy moment. The correction acts on the
distribution-function derivative at each electric push. It does not adjust a
completed step's total energy or add an energy-accounting ledger.

## Centered-stencil identity

Let $h=\Delta v$, $v_j=(j+\tfrac12)h$, $j=0,\ldots,N-1$, and
$v_g=v_{N-1}+h$. The current code uses odd parity for the lower $f_1$ ghost cell
and zero for the upper ghost cell. Define the real contraction

$$
s_j=E_x f_{1,j}^0+2\operatorname{Re}[(E_y+iE_z)f_{1,j}^1].
$$

The uncorrected isotropic electric rate is

$$
\dot f_{0,j}=\frac13\left[D_v s_j+\frac{2s_j}{v_j}\right].
$$

Summation by parts on this finite centered grid gives the exact discrete identities

$$
\dot n_e=\frac{2\pi}{3}v_g^2s_{N-1},
$$

$$
\dot{\mathcal E}_e=
\mathbf J\cdot\mathbf E
-\frac{8\pi}{3}h^2\sum_j v_js_jh
+\frac{\pi}{3}v_g^4s_{N-1},
\qquad
\mathbf J\cdot\mathbf E=-\frac{4\pi}{3}\sum_j v_j^3s_jh.
$$

The terms containing $s_{N-1}$ are the upper-tail contributions of this stencil.
They vanish for a resolved distribution whose tail is negligible. The middle
energy term is an interior discretization defect and persists even with a
vanishing tail. Tests with all electric-field components and both vanishing and
finite tails verify these identities directly, including independent harmonic
truncation.

## Local correction and supported scope

The added energy rate is the exact opposite of the interior defect:

$$
\Delta W=\frac{8\pi}{3}h^2\sum_j v_js_jh.
$$

A local density-neutral basis
$\psi_j=(v_j^2-\langle v^2\rangle)f_{0,j}$ realizes it through

$$
\dot f_{0,j}\mathrel{+}=
\frac{\Delta W\,\psi_j}{2\pi\sum_k v_k^4\psi_kh}.
$$

This is a distribution-dependent rank-one, $O(h^2)$ correction for a resolved
smooth distribution. It leaves the
density rate, higher harmonics, and explicit upper-tail terms unchanged. It uses
the same density-neutral moment helper as the ion-frame remap and local exchange.
A second centering pass suppresses density cancellation, and a variance-scaled
roundoff threshold rejects zero or degenerate radial energy responses. This
prevents a single occupied radial node from generating artificial density through
division by a rounding-only response. Physical use requires a nonnegative
distribution with resolved radial energy spread. Large explicit increments can
make $f_0$ negative; this correction supplies no positivity guarantee. Positivity
checks and temporal, radial, spatial, and angular convergence remain necessary.

`TzoufrasVlasov(..., conserve_electric_work=True)` is the nonrelativistic default.
Setting it to `False` retains the original operator for controlled comparisons.
The correction requires streaming speed equal to radial momentum, $v=p$ in the
solver's nonrelativistic normalization. Other streaming speeds are rejected when
the correction is enabled. `BaseVFP2D` explicitly disables it in relativistic
mode; a relativistic discrete energy correction is not implemented.

## Coupled verification and remaining errors

The [finite-field coupling diagnostic](moving_frame.md) is also run to $t=2$,
four times its original interval, on the same $12\times4$ grid. At $n_v=32$,
$\Delta t=0.01$, the accounted defect changes from **+7.3874%** with the original
stencil to **-0.05698%** with the work correction. The percentages use initial
transverse magnetic perturbation energy. Halving $\Delta t$ gives -0.05696%;
doubling $n_v$ to 64 gives -0.05379%. At that finer resolution the raw defect is
**+1.0008%**, with **+1.0545%** current-projection work recorded separately.
Quasineutrality remains below $3\times10^{-14}$ in these checks.

The declared regression requires less than 0.1% accounted defect at $t=2$ and
more than hundredfold improvement over the uncorrected control. It also bounds
the sensitivity to temporal and radial refinement. This interval is about 3.1%
of the guide-field Alfvén period, so it is not a full-period kinetic-wave or
experiment-duration validation. The ideal magnetic subsystem has a separate
full-period test.

The correction does **not** repair the electron momentum discretization. For an
isotropic distribution, for example, its electric-force moment remains

$$
\dot{\mathbf P}_e=-n_e\mathbf E
-\frac{4\pi}{3}h^2\left[\sum_j f_{0,j}h\right]\mathbf E
+\frac{2\pi}{3}v_g^3 f_{0,N-1}\mathbf E.
$$

A dedicated test preserves and exposes this residual. The ion magnetic source
uses $c^2(\nabla\times\mathbf B)\times\mathbf B$ directly, so the electron
force-moment defect does not enter that source. Electron current projection and
its lab-frame work remain separate limitations of the coupled kinetic method.

The earlier short, initially unmagnetized nonlinear check now has an accounted
residual near $9\times10^{-10}$ of its total-energy scale on both coarse and
fine grids. Its old radial/spatial error ratios no longer measure convergence of
the removed work defect. The updated test tightens its absolute residual bound
and limits refinement sensitivity; it does not identify the remaining residual's
cause. Longer flow-time convergence, angular refinement, and projection-free
energy closure remain open.
