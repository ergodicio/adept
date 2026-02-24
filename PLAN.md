Let's do some refactoring of the _vlasov1d module. This is going to be one step of a gradual refactoring
that replaces the "cfg"-based data threading in the code with a set of actual domain objects.

The way things currently work is that the ergoExo object calls a series of "lifecycle" methods on the ADEPTModule
(in this case, BaseVlasov1D). These modify the `self.cfg` field iteratively, adding stuff like normalization,
save/diagnostic callback functions, numpy arrays for the grid, etc., all onto that one dict.

We would like to move all of the simulation setup into the constructor of the simulation. We want a new class,
Vlasov1DSimulation, with roughly this constructor signature:
```
class Vlasov1DSimulation():
    def __init__(self, plasma_norm: PlasmaNormalization, x_grid: XGrid, species: dict[str, Species], v_grids: dict[str, VGrid]):
        ...
```

However, we can't just do away with the BaseVlasov1D class.
A wrinkle is that after each lifecycle call, the ergoExo wants to "log" the dictionary so far to an mlflow instance.
So together, the classes will work something like this:
```
def sim_from_cfg(cfg: dict) -> Vlasov1DSimulation:
    ...

class BaseVlasov1D(ADEPTModule):
    def __init__(self, cfg) -> None:
        self.simulation = sim_from_cfg(cfg)
        self.cfg = cfg

    def write_units(self) -> dict:
        self.cfg["units"]["derived"] = BaseVlasov1D.derived_units_dict(self.simulation.plasma_norm)
        self.cfg["grid"]["beta"] = self.simulation.plasma_norm.beta
        return self.cfg["units"]["derived"]

    def get_derived_quantities(self):
        self.cfg["terms"]["species"] = BaseVlasov1D.species_config_dict(self.simulation.species)
        self.cfg["grid"] = BaseVlasov1D.grid_scalars_dict(self.simulation.x_grid)
        ...
```

Essentially, to maintain backwards compatibility, we are going to construct the whole tree of domain
objects up front in the constructor, and then pretend that we're constructing it bit by bit in the
lifecycle methods.

## Already done
- We have added a regression test for the backwards compatibility of the config writing.

## Your task
The first class to introduce is Vlasov1DSimulation. What we'll do is have it gradually (commit by commit)
"eat up" more of the initialization of the BaseVlasov1D. In this step, we'll construct a PlasmaNormalization
and pass that to Vlasov1DSimulation. 

There is already an implementation of the PlasmaNormalization class and construction in the jj revision `twssnkrl` in the file vlasov1d/normalization.py

After this first step, the classes will look something like this:

```
class Vlasov1DSimulation():
    def __init__(self, plasma_norm: PlasmaNormalization):
        ...

def sim_from_cfg(cfg: dict) -> Vlasov1DSimulation:
    ...

class BaseVlasov1D(ADEPTModule):
    def __init__(self, cfg) -> None:
        self.simulation = sim_from_cfg(cfg)
        self.cfg = cfg

    def write_units(self) -> dict:
        self.cfg["units"]["derived"] = BaseVlasov1D.derived_units_dict(self.simulation.plasma_norm)
        self.cfg["grid"]["beta"] = self.simulation.plasma_norm.beta
        return self.cfg["units"]["derived"]

    def get_derived_quantities(self):
        # Same as now
        ...
```


