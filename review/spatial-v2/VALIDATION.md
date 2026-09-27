# Checkpoint validation — 2026-09-27

- Final `PYTHONPATH=vendor/ardy /Users/jakob/Desktop/Shellhacks/.venv/bin/python -m unittest discover -q`: **1,418 tests passed**, 57.668 seconds.
- `git diff --check`: passed.
- New API boundary test covers absent terrain fields, actor-specific origins, invalid actor IDs, non-finite/out-of-range heights.
- Terrain utility tests check actual feature masks and rigid native FK translation, exact unchanged non-Y channels, inverse transformation and input immutability.
- Optional terrain fitting and geometry tests include translated/yawed surfaces, missing bridge and actual door geometry.
- Offline contact experiment safety tests passed; **all three real motion candidates failed acceptance**. Passing software tests do not mean acceptable stair motion.
- 18 real native stair trials retained; none passed raw foot support. Full end-to-end terrain command/runtime path is not validated or integrated.
- Flat native door and courtyard generation, exact archive roundtrips and full-length browser captures completed; full temple traversal did not.
- No client files changed in this checkpoint. The preceding PR commit passed 45 client tests and production build; those are historical results, not rerun claims for this checkpoint.
- Latest runtime terrain-frame plumbing is CPU validated only; GPU trials used the standalone experimental driver. Run an isolated end-to-end runtime test before enabling it.
