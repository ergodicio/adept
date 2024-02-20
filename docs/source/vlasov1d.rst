Vlasov-Poisson 1D1V
=======================

To run the code for Vlasov-Poisson 1D1V, use the configs in `configs/vlasov1d/`.
There is a `mode` option in the config file which tells `ADEPT` which solver to use.

-----------------------
### Relevant features

1. Initialization - You can initialize the distribution function using a uniform or non-uniform density profile
2. Ponderomotive driver - You will need a driver to drive up a wave

Typical simulations
1. Single mode plasma waves (Landau damping, trapping)
2. Finite length plasma waves (Wavepackets)
3. Density gradients
4. Stimulated Raman Scattering
