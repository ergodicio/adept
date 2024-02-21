Vlasov-Poisson-Fokker-Planck 1D1V
====================================

To run the code for Vlasov-Poisson 1D1V, use the configs in `configs/vlasov1d/`.
There is a `mode` option in the config file which tells `ADEPT` which solver to use.

The normalized 1D1V Vlasov-Poisson-Fokker-Planck system is given by:

.. math:: 

    \frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x} + 
    E \frac{\partial f}{\partial v} = C_{ee}(f) + C_{K}(f)

    \partial_x^2 E = 1 - \int f dv

    C_{ee}(f) = \nu_{ee} \frac{\partial}{\partial v} 
    \left( v f + v_{th}^2 \partial_v f \right)

    C_K(f) = \nu_K (f - f_{Mx})

The ions are static but an enterprising individual should be able to reuse the electron code for the ions

**Things you might care about**

1. Infinite length (Single mode) plasma waves (Landau damping, trapping)
   
2. Finite length plasma waves (everything in 1. + Wavepackets)
   
3. Wave dynamics on density gradients (2 + density gradients)
   
4. Stimulated Raman Scattering (3 + light waves)


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

Collision frequency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
What is ``nu_ee``? It will modify the dynamics of the problem, possibly substantially depending on the distribution function dynamics. The envelope for this can also be specified
in the same way as the driver. 

Krook frequency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is another dissipative operator but in terms of physical correspondance, this mostly just resembles sideloss if anything. Use this as a hard thermalization operator, say for boundaries
as in the SRS example.

