# Data attribution & redistribution terms

**nucl-parquet's MIT license covers the code and the ENDF-6 → Parquet conversion only.**
The bundled evaluated nuclear data is third-party material. This file records,
per library, the custodian, redistribution terms, and the citation you must give.
The machine-readable source of truth is [`data/licenses.toml`](data/licenses.toml);
formal notices are in [`NOTICE`](NOTICE), and every subtree under `data/` carries
a generated `LICENSE.txt` sidecar.

> The tables below are generated. Edit `data/licenses.toml`, then run
> `python scripts/build_notices.py --write`.

**nucl-parquet and its downstreams (e.g. HYRR) are non-commercial academic
projects** of eXoma (ETH Zürich). The IAEA / JAEA / NDS terms below permit reuse
for research, education, and non-commercial products with acknowledgement, which
this redistribution falls under. **Commercial** users must obtain their own
permission from the relevant custodian for the IAEA- and JAEA-sourced libraries.

> Verdicts are from a primary-source audit (2026-06-01, every terms URL fetched
> directly), extended 2026-08-03 for trees that the first pass missed; those
> entries carry an `audit_basis` field recording how provenance was established.
> 🟢 clean · 🟡 redistributable with attribution / pending permission ·
> 🔴 not redistributable (removed).

## Summary

<!-- AUTO:libraries -->
| Library | Custodian | Terms | Cite |
|---|---|---|---|
| 🟢 BROND-3.1 | IPPE (Institute of Physics and Power Engineering), Obninsk | CC-BY-4.0 (site-wide) | A.I. Blokhin et al., 'New version of neutron evaluated data library BROND-3.1', Yad. Re… |
| 🟡 CENDL-3.2 | China Nuclear Data Center / CIAE | No written terms; NRDC open-mirror convention | Z. Ge et al., 'CENDL-3.2: The new version of Chinese general purpose evaluated nuclear… |
| 🟢 eXoma-computed derived tables (KERMA, neutron total/elastic, dose constants, spectrum-averaged XS) | Computed by exoma-ch from the bundled evaluated libraries | Computed output (data). Derived from ENDF/B-VIII.1, AME2020, ENSDF and NIST XCOM — the upstream attribution obligations flow through | nucl-parquet (eXoma, ETH Zürich) |
| 🟢 ENDF/B-VIII.0 | NNDC / Brookhaven National Laboratory / CSEWG (US DOE) | US Government work — public domain in the US (17 U.S.C. §105) | D.A. Brown et al., 'ENDF/B-VIII.0: The 8th Major Release of the Nuclear Reaction Data L… |
| 🟢 ENDF/B-VIII.1 | NNDC / Brookhaven National Laboratory / CSEWG (US DOE) | US Government work — public domain in the US (17 U.S.C. §105) | G.P.A. Nobre et al., 'ENDF/B-VIII.1: Updated Nuclear Reaction Data Library', Nuclear Da… |
| 🟡 EPDL97 / EADL / EEDL (EPICS — Evaluated Photon/Atomic/Electron Data Libraries) | D.E. Cullen (LLNL), distributed by the IAEA Nuclear Data Section | LLNL (US DOE) evaluation, openly distributed by IAEA-NDS; no CC license | D.E. Cullen, J.H. Hubbell, L. Kissel, 'EPDL97: the Evaluated Photon Data Library, '97 v… |
| 🟢 EXFOR (experimental reaction data) | International Network of Nuclear Reaction Data Centres (NRDC) / IAEA-NDS | CC-BY-4.0 (EXFOR Master File) | N. Otuka et al., 'Towards a More Complete and Accurate Experimental Nuclear Reaction Da… |
| 🟢 FENDL-3.2 | IAEA Nuclear Data Section | IAEA website terms grant non-commercial reuse with acknowledgement (commercial use gated); no CC license | 'FENDL: A library for fusion research and applications', Nuclear Data Sheets 193 (2024) 1 |
| 🟡 Geant4 G4EMLOW electron data (Seltzer-Berger brem DCS, electron stopping, density effect) | Geant4 Collaboration (CERN); underlying evaluation Seltzer & Berger (NIST) / ICRU-37 | Geant4 Software License v1.0 (NOT OSI) — notice + no-endorsement required | S.M. Seltzer, M.J. Berger, 'Bremsstrahlung energy spectra from electrons...', At. Data… |
| 🟡 Geant4 nuclear-structure data (G4ENSDFSTATE, PhotonEvaporation, RadioactiveDecay) | Geant4 Collaboration (CERN); underlying evaluation ENSDF (NNDC/IAEA-NDS) | Geant4 Software License v1.0 (NOT OSI) — notice + no-endorsement required | S. Agostinelli et al., 'Geant4 — a simulation toolkit', Nucl. Instrum. Meth. A 506 (200… |
| 🟢 HI-XS total reaction (Tripathi 1997) | Computed by exoma-ch (Tripathi 1997 parameterization) | Computed output (data) from a published parameterization | R.K. Tripathi, F.A. Cucinotta, J.W. Wilson, 'Accurate universal parameterization of abs… |
| 🟡 HI-XS Production (Geant4 INCL++/ABLA07) | Computed by exoma-ch with Geant4 (CERN); normalized to Tripathi (1997) | Computed output (data). Geant4 Software License v1.0 (NOT OSI) governs the generator | S. Agostinelli et al., 'Geant4 — a simulation toolkit', Nucl. Instrum. Meth. A 506 (200… |
| 🟡 IAEA-Medical | IAEA Nuclear Data Section (Coordinated Research Projects) | IAEA-NDS site copyright; per-sub-dataset citation; multi-institute contributions | Cite the specific sub-database evaluation paper(s) listed on each medical sub-page, plu… |
| 🟡 IAEA-PD-2019 (Photonuclear) | IAEA Nuclear Data Section | No license (GitHub data repo is NO-LICENSE); IAEA-NDS site copyright | T. Kawano, Y.S. Cho, P. Dimitriou et al., Nuclear Data Sheets 163 (2020) 109 |
| 🟡 IRDFF-II | IAEA Nuclear Data Section | IAEA-NDS site copyright (acknowledgment; 'no subsequent fee'); no CC license | A. Trkov, P.J. Griffin, S.P. Simakov et al., 'IRDFF-II: A New Neutron Metrology Library… |
| 🟢 JEFF-4.0 | OECD Nuclear Energy Agency (NEA) Data Bank | CC-BY-4.0 | Joint Evaluated Fission and Fusion Project (2025), JEFF-4.0 Evaluated Data, OECD Nuclea… |
| 🟡 JENDL-5 | Japan Atomic Energy Agency (JAEA), Nuclear Data Center | No explicit license on ENDF-6 source; copyright asserted, citation requested | O. Iwamoto et al., 'Japanese evaluated nuclear data library version 5: JENDL-5', J. Nuc… |
| 🟡 JENDL/AD-2017 | JAEA, Nuclear Data Center | No explicit license; copyright asserted | K. Shibata, N. Iwamoto, S. Kunieda, F. Minato, O. Iwamoto, 'Activation Cross-section Fi… |
| 🟡 JENDL/DEU-2020 | JAEA, Nuclear Data Center | No explicit license; copyright asserted | S. Nakayama, O. Iwamoto, Y. Watanabe, K. Ogata, 'JENDL/DEU-2020...', J. Nucl. Sci. Tech… |
| 🟡 NIST XCOM — X-ray mass attenuation coefficients | NIST (Physical Measurement Laboratory) | US public domain (17 U.S.C. §105) + NIST worldwide royalty-free reuse grant | M.J. Berger, J.H. Hubbell, S.M. Seltzer et al., XCOM: Photon Cross Section Database (ve… |
| 🟢 TENDL-2023 (+ Aug-2024 isomeric correction) | PSI — A.J. Koning, D. Rochman | No explicit license; openly distributed, citation requested | A.J. Koning, D. Rochman, J. Sublet, N. Dzysiuk, M. Fleming, S. van der Marck, 'TENDL: C… |
| 🟢 TENDL-2025 | PSI — A.J. Koning, D. Rochman | No explicit license; openly distributed, citation requested | D. Rochman, A. Koning, S. Goriely, S. Hilaire, 'TENDL-astro...', Nucl. Phys. A 1053 (20… |
| 🟢 Isotopic compositions / abundances | IUPAC (Commission on Isotopic Abundances and Atomic Weights) | Published reference data, citation requested | J. Meija et al., 'Isotopic compositions of the elements 2013', Pure Appl. Chem. 88 (201… |
| 🟢 Atomic masses / binding energies | Atomic Mass Data Center (AMDC) — AME2020 | Openly distributed evaluated data, citation requested | W.J. Huang, M. Wang, F.G. Kondev, G. Audi, S. Naimi, 'The AME 2020 atomic mass evaluati… |
| 🟡 NUDEX-derived statistical-model tables (level densities, PSF, ICC, capture gammas) | NUDEX (E. Mendoza et al., CIEMAT / IAEA CRP); underlying RIPL-3 and IAEA PSF database | IAEA-coordinated evaluated data, openly distributed; no CC license | E. Mendoza, D. Cano-Ott et al., NUDEX (nuclear de-excitation code) |
| 🟡 catima-computed stopping (catima_*.parquet) | Computed by exoma-ch with catima (A. Prochazka), based on ATIMA (GSI) | Computed output (data). catima itself is AGPL-3.0 (code) | J. Lindhard, A.H. Sørensen, Phys. Rev. A 53 (1996) 2443 |
| 🟡 PSTAR / ASTAR (+ derived dSTAR / tSTAR) and PSTAR/ASTAR compounds | NIST (Physical Measurement Laboratory) — SRD 124 | US public domain (17 U.S.C. §105) + NIST worldwide royalty-free reuse grant | M.J. Berger, J.S. Coursey, M.A. Zucker, J. Chang, ESTAR/PSTAR/ASTAR, NIST Standard Refe… |
<!-- /AUTO:libraries -->

🔴 **EAF-2010 was removed** (UKAEA licence forbids redistribution) — see issue #233.

## Copyleft-licensed tooling

Some tables are *computed by* copyleft-licensed programs. Copyleft attaches to
**distribution** of the program (GPL) or to **network interaction with a modified
version** (AGPL-3.0 §13) — **not** to "commercial use", and there is no academic
exemption. Data produced by running a program is not a derivative work of that
program, so the computed tables ship freely; what must not happen is shipping,
vendoring, or hard-depending on the copyleft code itself from an MIT
distribution.

<!-- AUTO:dependencies -->
| Dependency | SPDX | Role | Runtime dep? |
|---|---|---|---|
| [pycatima / catima](https://github.com/hrosiak/catima) | `AGPL-3.0-only` | build-time only — generates data/stopping/catima_*.parquet | **no** |
| [Geant4](https://geant4.web.cern.ch/download/license) | `LicenseRef-Geant4` | build-time only — generated hi-xs-prod; upstream of the G4EMLOW / nuclear-structure tables | **no** |
<!-- /AUTO:dependencies -->

## What you must do when redistributing

1. **Keep this attribution** (and `NOTICE`, and the per-directory `LICENSE.txt`)
   with the data — do not relicense the data as MIT.
2. **Cite** the libraries you actually use (citations above / in `licenses.toml`).
3. **Mark modifications** — the data was reformatted ENDF-6 → Parquet by eXoma
   (required by CC-BY-4.0 §3(a) and NIST terms).
4. **Pass attribution flow-down** for CC-BY data (EXFOR/JEFF/BROND) to your own
   downstream users.
5. **Carry the Geant4 notice** if you redistribute the ENSDF-derived structure
   tables, the EM electron tables, or `hi-xs-prod` — and do not imply Geant4
   endorsement.

## Permissions — optional for non-commercial use (tracked in #232)

As a non-commercial academic project we rely on the custodians' non-commercial /
open-distribution grants above; **none of these emails is required to ship.**
Drafts are kept for the record and for anyone who later needs a commercial grant:

- **IAEA-NDS** — *not needed* for non-commercial reuse (already granted with acknowledgement); send only if a commercial grant for FENDL/IRDFF/Medical/PD-2019/EPICS is ever required. Draft: [`docs/legal/permission-request-iaea.md`](docs/legal/permission-request-iaea.md)
- **JAEA** (`jendl@jaea.go.jp`) — *optional* insurance; JENDL has no explicit license either way, so redistribution rests on universal community practice. Draft: [`docs/legal/permission-request-jaea.md`](docs/legal/permission-request-jaea.md)
- **TENDL** (Koning/Rochman) — *courtesy* confirmation only. Draft: [`docs/legal/permission-request-tendl.md`](docs/legal/permission-request-tendl.md)
