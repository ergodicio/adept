Tests
=============

Run tests and examples
------------------------------
First install pytest via

.. code-block:: console

    (venv) $ pip install pytest
    (venv) $ pytest

This will run all the tests, which will likely include relatively expensive 2D2V Vlasov simulations.
If you only want to see example usage, you can choose particular tests by using the `-k` flag.

The package is tested against

1D1V Vlasov implementation
--------------------------------
- `test_landau_damping.py` - recover the real part and imaginary part (Landau damping) of the resoance according to the kinetic dispersion relation
- `test_absorbing_wave.py` - make sure the absorbing boundary conditions for the wave solver for the vector potential work correctly


1D two-fluid implementation
--------------------------------

- `test_resonance.py` - recover the Bohm-Gross dispersion relation and kinetic dispersion relation just using the forward pass

- `test_resonance_search.py` - recover the Bohm-Gross dispersion relation and kinetic dispersion relation using the backward pass

- `test_landau_damping.py` - recover the Landau damping rate according to the kinetic dispersion relation using a phenomenological term

- `test_against_vlasov.py` - recover a driven warm plasma wave simulation that is nearly identical to a Vlasov-Boltzmann simulation
