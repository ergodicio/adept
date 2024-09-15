Configuration Options
======================
``ADEPT`` needs a ``yaml`` file with the following datamodel to run the simulation. The datamodel is defined in the following class.

.. autoclass:: adept._lpse2d.datamodel.ConfigModel
    :members: __init__

Each of the objects used to initialize this datamodel can be treated just like dictionaries.  Each dictionary needs to be compiled into a megadictionary that is passed to the solver. 
The ``yaml`` configs accomplish this because a ``yaml`` is also a nested dictionary. The following documents those classes

High Level
-----------
These are the high level configuration options for the LPSE2D solver. Each of these either contains a fundamental type such as 
``bool``, ``int``, ``float``, or ``str`` or is another nested ``datamodel`` which can be treated just like a dictionary. 

.. autoclass:: adept._lpse2d.datamodel.UnitsModel
    :members: __init__

.. autoclass:: adept._lpse2d.datamodel.DensityModel
    :members: __init__

.. autoclass:: adept._lpse2d.datamodel.GridModel
    :members: __init__

.. autoclass:: adept._lpse2d.datamodel.SaveModel
    :members: __init__

.. autoclass:: adept._lpse2d.datamodel.MLFlowModel
    :members: __init__

.. autoclass:: adept._lpse2d.datamodel.DriverModel
    :members: __init__

.. autoclass:: adept._lpse2d.datamodel.TermsModel
    :members: __init__
    

Low Level
----------

The documentation for the nested datamodels is still TBD. To investigate them further, go to the source code.