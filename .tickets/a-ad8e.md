---
id: a-ad8e
status: closed
deps: [a-58a6]
links: []
created: 2026-01-06T22:18:21Z
type: feature
priority: 1
assignee: Jack Coughlin
---
# Phase 5: Update integrators for dict-based multi-species

Restructure time integrators to handle dicts of distributions with synchronized field solves.

## Design

Integrators receive and return `f_dict`. Since pushers and field solvers now handle dicts internally, integrator code is clean.

**LeapfrogIntegrator:**
```python
def __call__(self, f_dict, e_fields, prev_ex):
    f_dict = self.edfdv(f_dict, e_fields, 0.5*dt)  # Half step (all species)
    f_dict = self.vdfdx(f_dict, dt)                # Full step (all species)
    e = self.field_solver(f_dict, prev_ex, e_fields)  # Solve field (synchronized)
    f_dict = self.edfdv(f_dict, e, 0.5*dt)         # Half step (all species)
    return e, f_dict
```

**SixthOrderHamIntegrator:**
- Similar structure with multiple substeps
- Pushers handle species iteration internally
- Field solves happen at correct phase for all species simultaneously

Constructor updates:
- Pass `species_grids` and `species_params` to pushers and field solvers

## Files
- `adept/_vlasov1d/solvers/vector_field.py` (LeapfrogIntegrator, SixthOrderHamIntegrator)

## Acceptance Criteria
- Integrators handle dict input/output correctly
- Field solves synchronized across all species
- Leapfrog and 6th-order integrators both work
- Time integration preserves energy/conservation properties
