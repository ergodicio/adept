We are going to add multi-species (ions) support to the _vlasov1d module of this repository.

Your tasks:
1. Review the existing code structure in adept/_vlasov1d to understand the implementation
2. Form a plan in this file to add support for multiple species.

Key points:
- The implementation currently is driven by the vector_field.py -> VlasovPoissonFokkerPlanck and VlasovMaxwell classes.
    - These call into pushers for the vdfdx and edfdv actions; these can simply act on both distribution functions at once.
- The ubiquitous `f` argument can largely be changed to a pytree tuple of `(f_e, f_i, ...)`.
- We need to thread through the concept of multiple species' charge and mass (`q` and `m`) wherever `f` is currently passed
- The species will need to be defined in the configuration yaml files. Examples of these live in configs/_vlasov1d/....yaml
    - Note that the "species-" stanzas under `density:` are NOT what we want. These define different contributions to the _electron_ density.
    - Probably, we should put a new `species:` stanza under `terms:`. This can be a list of entries with a name, and q and m definitions.


## PLAN GOES HERE
