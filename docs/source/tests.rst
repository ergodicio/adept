Tests
=============
The package is tested against

1D two-fluid implementation
--------------------------------

- `test_resonance.py` - recover the Bohm-Gross dispersion relation and kinetic dispersion relation just using the forward pass

- `test_resonance_search.py` - recover the Bohm-Gross dispersion relation and kinetic dispersion relation using the backward pass

- `test_landau_damping.py` - recover the Landau damping rate according to the kinetic dispersion relation using a phenomenological term

- `test_against_vlasov.py` - recover a driven warm plasma wave simulation that is nearly identical to a Vlasov-Boltzmann simulation

2D Vlasov implementation
--------------------------------

- `test_landau_damping.py` - recover the Landau damping rate according to the kinetic dispersion relation using a phenomenological term


To run the tests
------------------
.. code-block:: console

    (venv) $ pip install pytest
    (venv) $ pytest
    (venv) $ pytest tests/<test_filename>