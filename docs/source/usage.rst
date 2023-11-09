Usage
=====

Installation
------------

To use adept, first install the requirements using pip:

.. code-block:: console

   $ python3 -m venv venv
   $ source venv/bin/activate
   (venv) $ pip install -r requirements.txt

or using conda:

.. code-block:: console

   $ mamba env create -f env.yaml
   $ mamba activate adept
   (adept) $


Run tests and examples
----------
First install pytest via

.. code-block:: console

    (venv) $ pip install pytest
    (venv) $ pytest

.. code-block:: console

    (adept) $ pip install pytest
    (adept) $ pytest

This will run all the tests, which will likely include relatively expensive 2D2V Vlasov simulations.
If you only want to see example usage, you can choose particular tests by using the `-k` flag.

