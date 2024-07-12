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

--------------


Run an example
--------------

The most common and obvious use case for ADEPT is a simple forward simulation that can be run from the command line. For example, to run a 1D1V Vlasov simulation of a driven electron plasma wave, use the following command:

.. code-block:: bash
    
    (venv) $ python3 run.py --cfg configs/vlasov-1d/epw

The input parameters are provided in `configs/vlasov-1d/epw.yaml`.  

**Access the output**

The output will be saved and made accessible via MLFlow. To access it, 

1. Launch an mlflow server via running ``mlflow ui`` from the command line
2. Open a web browser and navigate to http://localhost:5000
3. Click on the experiment name to see the results
