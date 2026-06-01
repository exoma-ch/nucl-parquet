# Data attribution & redistribution terms

**nucl-parquet's MIT license covers the code and the ENDF-6 → Parquet conversion only.**
The bundled evaluated nuclear data is third-party material. This file records,
per library, the custodian, redistribution terms, and the citation you must give.
The machine-readable source of truth is [`data/licenses.toml`](data/licenses.toml);
formal notices are in [`NOTICE`](NOTICE).

> Verdicts are from a primary-source audit (2026-06-01); every terms URL was
> fetched directly. 🟢 clean · 🟡 redistributable with attribution / pending
> permission · 🔴 not redistributable (removed).

## Summary

| Library | Custodian | Terms | Cite |
|---|---|---|---|
| 🟢 EXFOR | NRDC / IAEA-NDS | CC-BY-4.0 (Master File) | Otuka et al., NDS 120 (2014) 272 |
| 🟢 JEFF-4.0 | OECD/NEA | CC-BY-4.0 | JEFF Project (2025), DOI 10.82555/e9ajn-a3p20 |
| 🟢 ENDF/B-VIII.1 | NNDC/DOE | US public domain (17 USC §105) | Nobre et al., NDS 210 (2026) 1 |
| 🟢 BROND-3.1 | IPPE | CC-BY-4.0 | Blokhin et al. (2016) |
| 🟢 TENDL-2023-iso / 2025 | PSI (Koning/Rochman) | open, citation requested | Koning et al., NDS 155 (2019) 1 |
| 🟡 NIST PSTAR/ASTAR/ESTAR | NIST | US PD + worldwide grant; needs notices | Berger et al., SRD 124, DOI 10.18434/T4NC7P; ICRU 37/49 |
| 🟡 catima output | H. Rosiak / GSI ATIMA | computed data (code is AGPL — not shipped) | Lindhard-Sørensen (1996); Weaver et al. (2002) |
| 🟡 hi-xs-prod | Geant4 (CERN) | computed data; Geant4 notice required | Agostinelli et al., NIM A 506 (2003) 250 |
| 🟡 FENDL-3.2 | IAEA-NDS | no open license; commercial gated | NDS 193 (2024) 1 |
| 🟡 IRDFF-II | IAEA-NDS | site copyright | Trkov et al., NDS 163 (2020) 1 |
| 🟡 IAEA-Medical | IAEA-NDS | site copyright; per-sub-dataset cite | per sub-database paper |
| 🟡 IAEA-PD-2019 | IAEA-NDS | no repo license | Kawano et al., NDS 163 (2020) 109 |
| 🟡 JENDL-5 / AD-2017 / DEU-2020 | JAEA | copyright asserted; permission pending | Iwamoto et al., JNST 60 (2023) 1; + per-sublibrary |
| 🟡 CENDL-3.2 | CIAE/CNDC | no written terms; NRDC open-mirror | Ge et al., EPJ Web Conf. 239 (2020) 09001 |
| 🟡 ENSDF / AME2020 / IUPAC (meta) | NNDC, AMDC, IUPAC | open evaluated/reference data | ENSDF; Huang et al. (2021); Meija et al. (2016) |

🔴 **EAF-2010 was removed** (UKAEA licence forbids redistribution) — see issue #233.

## What you must do when redistributing

1. **Keep this attribution** (and `NOTICE`) with the data — do not relicense the
   data as MIT.
2. **Cite** the libraries you actually use (citations above / in `licenses.toml`).
3. **Mark modifications** — the data was reformatted ENDF-6 → Parquet by eXoma
   (required by CC-BY-4.0 §3(a) and NIST terms).
4. **Pass attribution flow-down** for CC-BY data (EXFOR/JEFF/BROND) to your own
   downstream users.

## Pending permissions (tracked in #232)
- **IAEA-NDS** — written clearance for FENDL/IRDFF/Medical/PD-2019 (esp. commercial bundling). Draft: [`docs/legal/permission-request-iaea.md`](docs/legal/permission-request-iaea.md)
- **JAEA** (`jendl@jaea.go.jp`) — JENDL-5 / AD-2017 / DEU-2020. Draft: [`docs/legal/permission-request-jaea.md`](docs/legal/permission-request-jaea.md)
- **TENDL** (Koning/Rochman) — courtesy confirmation. Draft: [`docs/legal/permission-request-tendl.md`](docs/legal/permission-request-tendl.md)
