"""Build-time converters for Geant4-derived nuclear data (epic #66, ADR-0002).

Each module under this package reads strata's published parquet
(``data/g4_raw/strata-nuclear/*.parquet``) and emits the schema-compatible
``meta/...`` outputs nucl-parquet ships, applying the unit/encoding
transforms specified in ADR-0002.

Per ADR-0002, transforms preserve the legacy column names (``level_keV``,
``half_life_s``, ``jp``) while dual-carrying strata-native columns
(``spin_x2``, ``parity``, ``floating_level_flag``, ``magnetic_moment_jt``,
``jpi``) for a future v1.0 cutover.

Modules
-------
- :mod:`nucl_parquet.g4.ensdfstate` — G4ENSDFSTATE3.0 → ``nuclides.parquet`` +
  ``ground_states.parquet``, joined with AME2020 (mass excess) and IUPAC
  (abundance) auxiliary tables (#69).
- :mod:`nucl_parquet.g4.photon_evap_levels` — PhotonEvaporation6.1.2 level
  schemes → per-element ``meta/ensdf/levels/{Symbol}.parquet`` (#70).
- :mod:`nucl_parquet.g4.radioactive_decay` — RadioactiveDecay6.1.2 →
  ``meta/decay.parquet`` + ``meta/decay_detailed.parquet`` (#71).
"""

from __future__ import annotations
