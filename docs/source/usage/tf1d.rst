Fluid-Poisson 1D
=======================

To run the code for TwoFluid-Poisson 1D, use the configs in `configs/tf-1d/`.
There is a `mode` option in the config file which tells `ADEPT` which solver to use.

The normalized 1D Fluid-Poisson system is given by:

.. note:: 

    Work in progress

.. math:: 

    \partial_t n_e + \partial_x (n_e u_e) = 0

    \partial_t u_e + u_e \partial_x u_e = -\frac{\partial_x P_e}{n_e}  - \frac{E}{n_e}

    \partial_x^2 E = n_i - n_e

The ions are static but all the functionality is in place if someone wants to get them to move!

**Things you might care about**

1. Infinite length (Single mode) plasma waves (Landau damping, trapping)
   
2. Finite length plasma waves (everything in 1. + Wavepackets)
   
3. Wave dynamics on density gradients (2 + density gradients)
   
4. Machine-learned fluid closures


-----------------------


Things you can change in the config file
----------------------------------------------

Density profile
^^^^^^^^^^^^^^^
Uniform is easy. For a non-uniform profile, you have to specify the parameters of the profile. 

The density profile can be parameterized as a sinusoidal perturbation or a tanh flat top. The parameters to the tanh flat-top are referred to in 

Ponderomotive Driver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You may want to a driver to drive up a wave. The envelope for this wave is specified via a tanh profile in space and in time. The other parameters to the wave
are the wavenumber, frequency, amplitude, and so on. Refer to the config file for more details
