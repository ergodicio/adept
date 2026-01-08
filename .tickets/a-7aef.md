---
id: a-7aef
status: open
deps: []
links: []
created: 2026-01-08T20:31:20Z
type: task
priority: 3
assignee: Jack Coughlin
---
# Normalize species config early to eliminate special-case code

Currently, the code has many special cases for "backward compatibility with single-species config files" throughout modules.py and helpers.py. This creates duplicate code paths and makes the codebase harder to maintain.

## Problem

When no species config is provided (single-species config files), the code branches into special backward-compatibility paths at multiple points:
- helpers.py:155 - entire else clause for backward compatibility
- modules.py:89-90 - grid-level dv computation
- modules.py:171-176 - fallback to grid-level values for nv/vmax

## Proposed Solution

Normalize the config early in the lifecycle (in `get_solver_quantities()` per modules.py:86 comment) by:
1. Check if `cfg["terms"]["species"]` exists
2. If not, generate a default species config for a single electron species:
   - name: "electron"
   - charge: -1.0
   - mass: 1.0
   - vmax: from cfg["grid"]["vmax"]
   - nv: from cfg["grid"]["nv"]
   - density_components: list of all keys in cfg["density"] that start with "species-"
3. Ensure this normalized config is available for all subsequent steps

## Benefits

- Single code path for both multi-species and single-species simulations
- Eliminates duplicate logic
- Easier to test and maintain
- Clear separation of concerns

## Implementation Notes

- The `species_found` check in helpers.py is still valuable and should be kept
- Refer to adept/_base_.py for lifecycle order (setup -> write_units -> get_derived_quantities -> get_solver_quantities)
- `get_solver_quantities()` is called early enough to normalize before other initialization

## Related Code Review Comments

- helpers.py:155: "This entire else clause can be avoided if we just provide a default species config"
- modules.py:86: "This is the appropriate place to normalize the species config stanza"
- modules.py:89-90: "Where are we using _this_ dv now at all? Can't we just rely on the species dv always?"
- modules.py:171: "remove this clause, there's no purpose for it"
