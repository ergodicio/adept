---
id: a-b69a
status: closed
deps: [a-8693]
links: []
created: 2026-01-06T22:18:20Z
type: feature
priority: 1
assignee: Jack Coughlin
---
# Phase 2: Multi-species initialization and data structures

Modify distribution initialization to return dict of species distributions with species-specific grids. Update state dictionary structure.

## Design

Modify `_initialize_distributions_()` in `helpers.py`:
- Return `dict[species_name, tuple[n_prof, f_s]]` instead of single distribution
- Each species has its own velocity grid based on config
- Sum density_components for each species separately

Update `init_state_and_args()` in `modules.py`:
- Create state dict with flat structure: `state[species_name] = f_s`
- Each `f_s` has shape `[nx, nv_s]` where `nv_s` is species-specific
- Quasineutrality: only set `ion_charge` if single species, else use zeros
- Build `species_grids` dict with grid parameters for each species
- Build `species_params` dict with q, m, qm for each species

## Files
- `adept/_vlasov1d/helpers.py`
- `adept/_vlasov1d/modules.py`

## Acceptance Criteria
- Multi-species state initializes with correct shapes for each species
- Backward compatibility: single-species configs still work
- Each species has independent velocity grid (different nv allowed)
- Quasineutrality handled correctly for single vs multi-species
