# X-ray + Auger synthesis from G4EMLOW × EC/IC vacancy fractions (issue #74)

| | |
|---|---|
| **Status** | Draft (v0.11.x) |
| **Date** | 2026-04-30 |
| **Phase** | H (issue #74) of epic #66 |
| **Related** | ADR-0002, issues #71 (per-shell EC), #72 (gammas / IC fractions) |

## Why a memo

Issue #74 is the heaviest physics task in the v0.11.0 G4 migration. Unlike #69–#73 (which are largely schema renames and unit conversions), this task derives X-ray and Auger emission rows from **three independent data sources** convolved through atomic-physics relations. It is the place where the migration most plausibly introduces a quiet correctness regression. This memo records the synthesis flow, the data sources, the approximations, and the known gaps so a reviewer can audit each step.

## Inputs

1. **Per-shell electron-capture (EC) fractions** — `data/meta/decay_detailed.parquet`
   (produced by `nucl_parquet/g4/radioactive_decay.py`, #71).
   Strata's RadioactiveDecay6.1.2 deliberately splits EC by atomic shell, exposing one row per `decay_mode ∈ {KshellEC, LshellEC, MshellEC, NshellEC}` per (parent, daughter level). Branching ratio for that shell × level is in `branching`.

2. **Gamma transitions + total internal-conversion coefficients** — `data/g4_raw/strata-nuclear/photon_evap_gammas.parquet` (strata's PhotonEvaporation6.1.2 export).
   Each gamma carries `intensity` (per-100 from origin level) and `icc_total` (the total internal-conversion coefficient α_T = N_e / N_γ). **Per-shell IC coefficients (α_K, α_L, …) are NOT exposed by strata** — only the summed `icc_total`. See "Known gaps" below.

3. **Atomic-relaxation transition probabilities** — `data/meta/eadl/{Symbol}.parquet`
   already shipped under nucl-parquet's `data/`. Schema: `(Z, vacancy_shell, filling_shell, transition_type ∈ {radiative, auger}, energy_keV, probability, edge_keV)`.

   * **Radiative rows** (vacancy K, filling L2/L3/M2/M3/…) carry the X-ray energy of the emitted photon (binding energy difference ≈ filling-shell − vacancy-shell, after relaxation corrections). The probability is normalized so that *radiative + auger* probabilities for a given vacancy sum to ≈ 1.0.
   * **Auger rows** in EADL are two-shell-tagged (vacancy + filling), but a real Auger transition is three-shell (vacancy, filling, ejected-from). The third shell varies across rows of the same `(vacancy, filling)` group, distinguishable only by `energy_keV`. We aggregate to canonical Auger groups (KLL, KLM, KMM, LMM, …) by mapping `(vacancy, filling)` to its first letter (K → "K", L1/L2/L3 → "L", M1–M5 → "M", …) and deriving the third letter from a heuristic (see below).

4. **State assignment** — `data/meta/ensdf/nuclides.parquet` (#69) is consulted to map `parent_ex_kev` → `state ∈ {"", "m", "m2", …}` using the existing ±1 keV fuzzy-match convention from `radioactive_decay.py`.

## Synthesis flow

For each parent `(Z, A, state)`:

### Step 1 — EC vacancy rate per shell (daughter Z' = Z − 1)

```
v_EC(K) = Σ branching(KshellEC, daughter_level)
v_EC(L) = Σ branching(LshellEC, daughter_level)
v_EC(M) = Σ branching(MshellEC, daughter_level)
v_EC(N) = Σ branching(NshellEC, daughter_level)
```
The sum is over all daughter excited levels. Note: G4 does not split L into L1/L2/L3 in `radioactive_decay.parquet`. We attribute the total `LshellEC` to vacancy_shell `"L1"` for relaxation lookup (dominant per atomic-physics: ~80–90% of L-shell EC populates L1 by selection rules). Same convention for M → M1, N → N1. **This is a documented approximation.**

### Step 2 — IC vacancy rate per shell (daughter Z' = Z, same atom)

The IC computation operates on isomer parents only (`parent_ex_kev > 0`, IT-decaying); ground-state EC parents whose daughters subsequently emit IC-converted gammas are a v0.12 follow-up (item 4 below).

For each isomer parent we:

1. **Resolve the isomer's level_idx** from `photon_evap_levels` by ±1 keV match against `parent_ex_kev`.
2. **Propagate cascade populations** down through the level scheme starting at `population[isomer_idx] = 1`. At each step the source level's population is split across daughter levels weighted by the *transition* branch fraction (i.e. `intensity × (1 + icc_total)` renormalized within the source level — using transitions, not just emitted gammas, since IC and gamma are alternatives drawing from the same pool).
3. **Sum K-vacancies per transition**, gating to gammas whose energy exceeds the daughter K-edge:
   ```
   v_K = Σ_g (E_g ≥ E_K) × pop[source(g)] × transition_frac(g) × icc_total(g) / (1 + icc_total(g))
   ```

This is the canonical Tc-99m physics: the depopulating 2.17 keV gamma of the isomer is *below* the Tc K-edge (21 keV) so it creates no K-vacancies on its own; instead, that gamma populates level 1 (140.5 keV), and the *level-1 → ground* transition (140.5 keV gamma, α_T = 0.113) is what drives K-vacancy production. Without cascade propagation the synthesizer underestimates Tc-99m K-vacancy yield by ~10× (drops from canonical 10% to ~0.9%).

**Why K-shell only:** strata's `photon_evap_gammas.parquet` only ships `icc_total`, not the per-shell partials α_K / α_L1 / α_L2 / …. The raw G4 PhotonEvaporation files contain the partials but they have not been carried through to the strata export at the pinned revision (catalog SHA `9a74e823…`). For low-Z parents (Z < 30) K-shell IC dominates (α_K / α_T > 0.85 typically); for medium-Z (30 ≤ Z < 60) it remains the largest single contributor; for high-Z (Z ≥ 60) L-shell IC can rival or exceed K. The approximation therefore degrades with Z.

**Mitigation in v0.11.x:** documented as a known-gap follow-up (filed as a sub-issue of #74). v0.11.0 ships the K-only approximation; a future PR enriches strata to carry the partials and revisits this module.

**Vacancy cap:** `vacancy_rate` is clipped at 1.0 (one K-vacancy per parent decay is the physical maximum). Numerical / branching-renormalization noise can push slightly above unity for some entries; the clip ensures downstream consumers never see > 100% intensity from a single shell.

### Step 3 — Look up EADL transitions for each vacancy

For each `(daughter_Z, vacancy_shell)` pair where `v_total > 0`:

* **X-ray rows** — one per radiative EADL entry:
  `intensity_pct = v_total × probability × 100`
  `rad_subtype = canonical_xray_label(vacancy, filling)` (e.g. K→L3 = "Kα1", K→L2 = "Kα2", K→M2 = "Kβ3", K→M3 = "Kβ1", K→N2/N3 = "Kβ2", L3→M5 = "Lα1", L3→M4 = "Lα2", L2→M4 = "Lβ1", …).

* **Auger rows** — aggregated by `(vacancy_first_letter, filling_first_letter, third_letter)`:
  We group EADL rows by the two-letter code, then by inferring the third shell from the energy: a KLL Auger has energy ≈ E_K − 2·E_L; a KLM has energy ≈ E_K − E_L − E_M; etc. We **don't** try to disaggregate the third shell — instead we emit one row per *unique energy* and label it by the dominant Auger group from the (vacancy, filling) pair, e.g. all `(K, L?)` rows are labelled `"KLL"` if they're below the K-edge minus an L-binding, otherwise `"KLM"`. This matches the granularity of v0.10.x's `"Auger K"` / `"Auger L"` rad_subtype but with finer line-by-line resolution.

### Step 4 — Energy-conservation invariant

For each parent and each shell vacancy, the EADL rows satisfy
`Σ probability(radiative) + Σ probability(auger) ≈ 1.0`
(verified at module load — small element-dependent rounding errors of ≤ 0.5 % are common in the EADL release).

We therefore enforce a sweep test:
```
for each (Z, A, state):
    for each shell vacancy in daughter:
        |Σ intensity_pct(xray, this vacancy) + Σ intensity_pct(auger, this vacancy) − v_total × 100| ≤ 5 %
```
The 5 % slack absorbs:
* EADL probability rounding (~0.5 %)
* Mapping-to-L1-only approximation (typically < 5 % for low-Z, can rise for high-Z)
* Coster–Kronig vacancy transfer not yet folded in (see "Known gaps" below)

## Known gaps (logged as v0.12 follow-ups)

1. **Per-shell IC coefficients absent from strata** — see Step 2. K-only approximation, accuracy degrades with Z. Action: enrich strata's PhotonEvaporation export with `icc_K, icc_L1, icc_L2, icc_L3, icc_M1…M5, icc_outer`.

2. **L1-only / M1-only mapping for EC L/M/N branchings** — the underlying physics splits these by sub-shell (L1, L2, L3 etc.) but our input only ships the shell totals. Sub-shell branchings come from atomic capture probabilities (P_L1 / P_L = ~0.8 for medium-Z, weakly Z-dependent). v0.11 attributes everything to L1 / M1 / N1 — biases X-ray output toward Lβ over Lα by a small amount.

3. **Coster–Kronig vacancy transfer** — when an L1 vacancy is filled by an L2/L3 electron (intra-shell radiationless transition), it transfers the vacancy to L2/L3 which then relaxes. EADL includes some of these transitions (e.g. L1 → L2 + Auger from M, or L1 → L3 + Auger from M) but the cascade isn't *re-entered* in our convolution: we treat each shell vacancy independently. Net effect: under-counts L2/L3-filling X-rays relative to a full cascade simulation. Acceptable for v0.11.x dose-calc use cases; would need a Monte-Carlo cascade for spectroscopy-grade accuracy.

4. **EC-then-IC chains** — Step 2 currently considers only the parent's own state's gamma cascade for IC. Daughter-level cascades populated by EC are *not* mapped to additional IC vacancies. For Co-57 (which feeds Fe at 136/14 keV levels via EC, then the 14 keV gamma is mostly IC-converted), the IC X-rays from Fe are *not* present in the v0.11.0 output — only the EC X-rays are. This is the largest accuracy gap for v0.11.0 and is logged as a v0.12 enrichment.

5. **Daughter-level state inheritance** — IT decay from `(Z, A, m)` produces a daughter at the parent's lower levels (same Z), whose subsequent gammas may be IC-converted. We handle this by reading `photon_evap_gammas` for `(Z, A)` and selecting transitions whose origin level is ≤ the parent isomer's `parent_ex_kev` — for Tc-99m → Tc-99 (level 1, 140.5 keV), this captures the M1 transition correctly.

## Integration contract — `radiation_atomic/` → `radiation/` merge (#75)

The v0.11 X-ray + Auger synthesis ships to `data/meta/ensdf/radiation_atomic/{Symbol}.parquet` rather than directly into the canonical `data/meta/ensdf/radiation/{Symbol}.parquet`. The validation harness (#75) must merge them into the canonical layout before v0.11.0 ships. Contract:

- **Union by row, not column**: `radiation_atomic/{Symbol}.parquet` rows have `rad_type IN ('xray', 'auger')`; `radiation/{Symbol}.parquet` rows from #72 have `rad_type = 'gamma'`. The merge concatenates by row across both sources, preserving `rad_type` as the discriminator.
- **Schema reconciliation**: `radiation_atomic` columns include `parent_level_keV` and `rad_subtype` (e.g. `"K-L3"`, `"KLL"`); `radiation` already carries those columns post-#72. Bonus columns specific to one source (`multipolarity`, `mixing_ratio`, `icc_total`, `daughter_level_keV` from #72; `vacancy_shell` from this PR) get NULL on the other source's rows.
- **Uniqueness**: the union must not produce duplicate `(Z, A, state, rad_type, energy_keV, rad_subtype)` rows — assert this in #75. Identical `(Z, A, state)` may appear on both sources but must differ in `rad_type`.
- **State assignment**: both sources derive `state` from the nuclides catalog (#69). #75 should re-derive consistently from the post-merge catalog if any drift is observed.

After #75 merges, `data/meta/ensdf/radiation_atomic/` is removed (per `git rm`) and the canonical `data/meta/ensdf/radiation/{Symbol}.parquet` carries every emission row from any source.

## Acceptance spot-checks (issue #74) — observed values vs canonical

| Test | Observed | Canonical (literature) | Source |
|---|---|---|---|
| Tc-99m Kα1 (Tc daughter from IC) | 18.33 keV @ 4.69 % | 18.37 keV @ 4.36 % | EADL Tc, K→L3 |
| Tc-99m Kα2 | 18.21 keV @ 2.47 % | 18.25 keV @ 2.34 % | EADL Tc, K→L2 |
| Tc-99m Kβ1 | 20.59 keV @ 0.75 % | 20.62 keV @ 0.85 % | EADL Tc, K→M3 |
| Co-57 EC → Fe Kα1 | 6.36 keV @ 17.6 % | 6.40 keV @ ~16.5 % | EADL Fe, K→L3 (Mössbauer line) |
| I-125 EC → Te Kα1 | 27.47 keV @ 37.6 % | 27.47 keV @ ~38 % | EADL Te, K→L3 |
| I-125 KLL Auger sum | 9.42 % | ~9.5 % | EADL Te, K-vacancy auger rows |
| Co-57 Σ(K X-ray + K Auger) | 88.77 % | 88.77 % (= K-EC fraction × 1.0) | Energy-conservation invariant |
| Tc-99m Σ(K X-ray + K Auger) | 10.93 % | ~10.5 % | Energy-conservation invariant (cascade) |

## References

- ADR-0002 — schema decision (`docs/adr/0002-g4-migration-schema.md`)
- ICRP Publication 107 — Nuclear Decay Data for Dosimetric Calculations (canonical isotope set)
- Geant4 `G4AtomicTransitionManager` — reference implementation of the same convolution
- EADL: Perkins, Cullen, Chen, Hubbell, Rathkopf, Scofield, *Tables and Graphs of Atomic Subshell and Relaxation Data Derived from the LLNL Evaluated Atomic Data Library (EADL)*, UCRL-50400 vol. 30, LLNL (1991).
- Strata published dataset: `gerchowl/strata-data` on Hugging Face, catalog pin `9a74e823…`.
