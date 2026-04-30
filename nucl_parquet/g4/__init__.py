"""G4-derived data converters.

Build-time scripts that transform Geant4 ASCII data files (already published as
Parquet by the strata project) into nucl-parquet's v0.10.x-compatible parquet
schemas. Per ADR-0002, transforms preserve the legacy column names (`level_keV`,
`half_life_s`, `jp`) while dual-carrying strata-native columns (`spin_x2`,
`floating_level_flag`, `magnetic_moment_jt`) for a future v1.0 cutover.

Modules
-------
- :mod:`nucl_parquet.g4.ensdfstate` — G4ENSDFSTATE3.0 → ``nuclides.parquet`` +
  ``ground_states.parquet``, joined with AME2020 (mass excess) and IUPAC
  (abundance) auxiliary tables.
"""

from __future__ import annotations
