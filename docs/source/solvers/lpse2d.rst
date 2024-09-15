Enveloped equations in Cartesian 2D
====================================



Equations and Quantities
-------------------------
These equations model the evolution and interaction of the complex envelopes of light waves and plasma waves. This is faster than modeling the plasma waves using a fluid or kinetic solver along with modeling the light waves

Note on pump depletion
^^^^^^^^^^^^^^^^^^^^^^^^
One can solve these equations with or without "pump depletion". "Pump depletion" is the effect of the plasma waves on the light waves. We do not currently have this implemented, so we have light waves that behave as external drivers for the plasma waves and we only model the plasma wave response. 
This approach is adequate for modeling laser plasma instabilities below the absolute instability threshold.

Electron Plasma Waves
^^^^^^^^^^^^^^^^^^^^^^^

.. math::
    \nabla \cdot \left[ i \left(\frac{\partial}{\partial t} + \nu_e^{\circ} \right) + \frac{3 v_{te}^2}{2 \omega_{p0}} \nabla^2 + \frac{\omega_{p0}}{2}\left(1-\frac{n_b(x)}{n_0}\right) \right] \textbf{E}_h = S_{TPD} + S_h


Two Plasmon Decay
^^^^^^^^^^^^^^^^^^^^

.. math::
    S_{\text{TPD}} \equiv \frac{e}{8 \omega_{p0} m_e} \frac{n_b(x)}{n_0} \nabla \cdot [\nabla (\textbf{E}_0 \cdot \textbf{E}_h^*) - \textbf{E}_0 \nabla\cdot \textbf{E}_h^*] e^{-i (\omega_0 - 2 \omega_{p0})t}

Laser Driver
^^^^^^^^^^^^^^^
We only have a plane wave implementation for now

.. math::
    E_0(t, x, y) = \sum_j^{N_c} A_j ~ \exp(-i k_0 x - i \omega_0 \Delta\omega_j ~ t + \phi_j)


Configuration Options
----------------------

As with all other solvers, the configuration is passed in via a ``yaml`` file. The datamodel for the configuration is documented below

.. toctree::
    datamodels/lpse2d
    :maxdepth: 3
    :caption: Configuration Options: