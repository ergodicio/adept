---
id: a-d5c9
status: open
deps: []
links: []
created: 2026-01-08T20:28:42Z
type: task
priority: 2
assignee: Jack Coughlin
---
# Support multi-species distributions in diagnostics

Currently, diagnostics use a reference distribution from the first species only (modules.py:257). This will NOT work correctly when running multiple species simulations.

## Tasks

- Store species distributions separately for diagnostics
- Update diagnostic calculations to handle multi-species data
- Ensure diagnostic outputs properly label and track each species

## Notes

Related to multi-species support implementation. This issue was identified during code review.
