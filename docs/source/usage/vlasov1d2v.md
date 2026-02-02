# Vlasov-Poisson-Fokker-Planck 1D2V

To run the code for Vlasov-Poisson 1D2V, use the configs in `configs/vlasov1d/`.
There is a `mode` option in the config file which tells ADEPT which solver to use.

## Equations

The normalized 1D2V Vlasov-Poisson-Fokker-Planck system is given by:

$$
\frac{\partial f}{\partial t} + v_x \frac{\partial f}{\partial x} + E_x \frac{\partial f}{\partial v_x} = C_{ee}(f) + C_{K}(f)
$$

$$
\partial_x^2 E = 1 - \int f \, dv
$$

where the collision operators are:

$$
C_{ee}(f) = \nu_{ee} \frac{\partial}{\partial \mathbf{v}} \cdot \left( \mathbf{v} f + v_{th}^2 \partial_\mathbf{v} f \right)
$$

$$
C_{ei}(f) = \nu_{ei}(v) \left[\partial_{v_x} \left(-v_y^2 \partial_{v_x} f + v_x v_y \partial_{v_y} f\right) + \partial_{v_y} \left(v_x v_y \partial_{v_x} f - v_x^2 \partial_{v_y} f\right)\right]
$$

$$
C_K(f) = \nu_K (f - f_{Mx})
$$

The ions are static but an enterprising individual should be able to reuse the electron code for the ions.

## Things You Might Care About

1. Infinite length (Single mode) plasma waves (Landau damping, trapping)
2. Finite length plasma waves (everything in 1. + Wavepackets)
3. Wave dynamics on density gradients (2 + density gradients)

---

## Configuration Options

### Density Profile

Uniform is easy. For a non-uniform profile, you have to specify the parameters of the profile.

The density profile can be parameterized as a sinusoidal perturbation or a tanh flat top. See [initialization](initialization.md) for details on the tanh flat-top parameters.

### Ponderomotive Driver

You may want a driver to drive up a wave. The envelope for this wave is specified via a tanh profile in space and in time. The other parameters to the wave are the wavenumber, frequency, amplitude, and so on. Refer to the config file for more details.

### Collision Frequency

What are `nu_ee` and `nu_ei`? They will modify the dynamics of the problem, possibly substantially depending on the distribution function dynamics. The envelope for this can also be specified in the same way as the driver.

### Krook Frequency

This is another dissipative operator but in terms of physical correspondence, this mostly just resembles sideloss if anything. Use this as a hard thermalization operator, say for boundaries as in the SRS example.
