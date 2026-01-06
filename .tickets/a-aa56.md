---
id: a-aa56
status: open
deps: [a-419a]
links: []
created: 2026-01-06T22:18:21Z
type: feature
priority: 1
assignee: Jack Coughlin
---
# Phase 7: Update storage and diagnostics for multi-species

Generalize NetCDF storage and diagnostics to handle multiple species with different grid sizes.

## Design

**storage.py changes:**

`store_f()`:
- Change signature to accept `species_names` parameter
- Loop over species_names instead of hardcoded ["electron"]
- Save each species distribution: `f_{species_name}[nt, nx, nv_s]`

`get_dist_save_func()`:
- Read species list from `cfg.terms.species`
- Create save functions for each species in config
- Handle species-specific save intervals (from `save.{species_name}.t`)

**modules.py changes:**
- Update `BaseVlasov1D.save()` to pass `species_names` to storage functions
- NetCDF variables include species name in variable naming

**Diagnostics:**
- Generalize moment calculations (density, temperature, etc.) per species
- Store per-species diagnostics in output files

## Files
- `adept/_vlasov1d/storage.py`
- `adept/_vlasov1d/modules.py`

## Acceptance Criteria
- Multi-species results save correctly to NetCDF
- Each species can have different save intervals
- Distributions with different nv_s save correctly
- Can load and visualize multi-species results
- Backward compatible: single-species saves work unchanged
