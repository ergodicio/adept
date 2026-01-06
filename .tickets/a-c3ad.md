---
id: a-c3ad
status: open
deps: [a-7503]
links: []
created: 2026-01-06T22:18:21Z
type: chore
priority: 2
assignee: Jack Coughlin
---
# Phase 9: Documentation

Document new multi-species configuration schema and provide usage examples.

## Documentation Tasks

1. **Configuration Schema**:
   - Document new `terms.species` section with all fields
   - Explain species-specific velocity grids (vmax, nv)
   - Provide examples with electrons + ions
   - Document backward compatibility behavior

2. **Example Configurations**:
   - Ion acoustic wave
   - Electron-ion two-stream
   - Landau damping with ions
   - Include comments explaining parameters

3. **API Documentation**:
   - Update docstrings for modified functions
   - Document state dictionary structure
   - Explain species_grids and species_params dicts

4. **User Guide**:
   - How to set up multi-species simulation
   - Choosing appropriate velocity grids per species
   - Typical q, m values for different ion species
   - Normalization conventions (everything in electron units)

5. **Migration Guide**:
   - How existing configs continue to work
   - How to add ions to existing electron simulation
   - Performance considerations (different nv per species)

## Files
- README or docs/ directory
- Example configs in `configs/vlasov-1d/`
- Docstrings in modified Python files

## Acceptance Criteria
- Complete configuration schema documented
- At least 3 example multi-species configs provided
- Users can set up multi-species simulation from docs
- Migration guide helps existing users adopt new features
