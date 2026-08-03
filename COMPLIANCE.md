# Compliance record

> **Status: DRAFT determinations — pending ETH legal / Technology Transfer sign-off.**
> Not legal advice. Tracks issues #238 (export control) and #239 (IP / release rights).

**Project status:** nucl-parquet and its downstreams (e.g. HYRR) are
**non-commercial academic** projects of eXoma (ETH Zürich). This is the basis for
the data-reuse posture in [`ATTRIBUTION.md`](ATTRIBUTION.md): the custodians'
non-commercial / open grants cover this redistribution with acknowledgement.

## 1. Export control / dual-use (#238)

**Determination (to be confirmed by ETH):** redistributing the bundled
**published, evaluated nuclear data** is covered by the *"in the public domain"*
and *"basic scientific research"* carve-outs of:
- EU Dual-Use Regulation **(EU) 2021/821** (General Technology Note), and
- the Swiss **Güterkontrollverordnung (GKV)** / Goods Control Act, which is
  harmonised with the same framework.

This includes the **Russian-origin (BROND-3.1 / IPPE)** and **Chinese-origin
(CENDL-3.2 / CIAE)** libraries: re-publishing already-public scientific data from
a Swiss/EU entity does not breach export-control or sanctions law (no transaction
with the entity; published data is exempt). Residual concern is reputational, not
legal.

The one library NOT covered — **EAF-2010 (UKAEA)** — has been **removed** (its
licence forbade redistribution; UKAEA fusion-activation data may also touch UK
export sensitivities). See #233.

This determination also satisfies **RSETHZ 440.4 Art. 27(1)(b)**, which makes
export-control compliance an explicit precondition of any ETH open-source
release.

## 2. IP / open-source release rights (#239)

**Determination (to be confirmed by ETH Transfer):** eXoma holds the right to
release nucl-parquet's **code and the ENDF-6→Parquet conversion** under MIT, and
to redistribute the third-party data under the per-library terms in
[`ATTRIBUTION.md`](ATTRIBUTION.md).

Under **RSETHZ 440.4 Art. 5(3)** the exclusive economic exploitation rights in
software written by ETH employees on duty belong to **ETH Zürich**; the moral
rights stay with the creators. The copyright holder string is therefore
**"ETH Zürich (eXoma — Exotic Matter Applications)"**, with the creators named
alongside it, as **Art. 27(1)(e)** requires.

## 3. ETH Zürich RSETHZ 440.4 — open-source release conditions

[RSETHZ 440.4](https://ethz.ch/content/dam/ethz/main/eth-zurich/organisation/rechtssammlung/440.4.pdf)
(*IP- und Verwertungsreglement*, in force 1 July 2026) permits ETH professors and
employed researchers to release software they created **solely and entirely
themselves** as open source, provided all conditions of Art. 27(1) hold. Status:

<!-- AUTO:eth -->
| Art. 27(1) condition | Status |
|---|---|
| (a) Release decided by the responsible Forschungsgruppenleiter:in | ☐ TBD — record the responsible Forschungsgruppenleiter:in (decision TBD) |
| (b) Complies with applicable law incl. export control | ☑ see COMPLIANCE.md §1 |
| (c) No conflict with ETH/third-party IP; dependency licences compatible | ☑ see COMPLIANCE.md §3 |
| (d) Published without delay on a public platform | ☑ github.com/exoma-ch/nucl-parquet |
| (e) Carries the OSS licence and © notice naming ETH Zürich + creators | ☑ LICENSE, NOTICE (holder: ETH Zürich) |
| Art. 27(3) — no CLA without ETH transfer approval | ☑ policy: `prohibited-without-eth-transfer-approval` |
| Art. 24 — software disclosure to ETH transfer | — not due (non-commercial) |
| Art. 34 — research-data disclosure to ETH transfer | — not due (non-commercial) |
<!-- /AUTO:eth -->

**Open items:**

- **Art. 27(1)(a) — the release decision is the research group leader's, not the
  creators'.** It must be recorded. Set `group_leader` and `oss_decision_date` in
  [`data/licenses.toml`](data/licenses.toml) `[eth]` and flip
  `oss_decision_recorded = true`. If several group leaders are involved the
  decision must be **unanimous**; deadlock or conflict of interest escalates to
  the VPWW. ETH transfer publishes an optional *Acknowledgement Form of OSS
  Distribution and License* (footnote 21 of the regulation) — recommended for
  documentation, not required.
- **Art. 27(1)(a) scope caveat.** The Art. 27 permission is written for software
  created *solely and entirely* by ETH researchers. nucl-parquet's own code
  qualifies; the bundled third-party data does not travel under Art. 27 at all —
  it travels under the custodian terms in `ATTRIBUTION.md`. Keeping the MIT grant
  scoped to code-and-conversion is what keeps those two régimes separate.

### Art. 27(1)(c) — dependency licence compatibility

Art. 27(1)(c) forbids releasing software that includes or links third-party code
unless every part of it is under an OSS licence **compatible with the intended
outbound licence**. Because the outbound licence here is MIT (permissive), any
copyleft dependency would be incompatible if it were linked or shipped.

Current posture, generated into `ATTRIBUTION.md` from
`[code_dependencies]` in the manifest:

- **`pycatima` / catima is AGPL-3.0** and must stay a **build-time** dependency.
  It generates `data/stopping/catima_*.parquet`; the runtime loader reads those
  Parquet shards and never imports it. Declaring it in `[project].dependencies`
  would install an AGPL-3.0 library into every user's environment alongside an
  MIT wheel — incompatible under Art. 27(1)(c) and contradicting `NOTICE`.
- **Geant4** is not OSI-licensed. No Geant4 source or binary is redistributed;
  only computed output and reformatted data tables, with the mandatory notice and
  no-endorsement statement carried in `NOTICE`.

Copyleft is triggered by **distribution** (GPL) or by **network interaction with
a modified version** (AGPL-3.0 §13) — not by "commercial use", and there is no
academic exemption. Purely internal use inside ETH, with no distribution outside
the legal entity, does not trigger either.

### Art. 27(3) — no CLAs without ETH transfer approval

ETH staff may not sign Contributor Licensing Agreements or similar third-party
agreements covering software created on duty without prior ETH transfer
approval. Recorded in [`CONTRIBUTING.md`](CONTRIBUTING.md); the project uses the
DCO instead of a CLA, which does not transfer rights and so is unaffected.

### Art. 24 / Art. 34 — disclosure duties

Both are triggered by intended **commercial exploitation** only, with a
**two-month** lead time to ETH transfer:

| Duty | Trigger | Status |
|---|---|---|
| Art. 24 — "Software Disclosure" | ≥2 months before planned commercial exploitation of the software | not due — non-commercial |
| Art. 34 — research data / databases | ≥2 months before planned commercial exploitation of the data | not due — non-commercial |
| Art. 19 — "Invention Disclosure" | ≥2 months before publishing anything patentable | n/a — no invention claimed |

If commercialisation is ever contemplated (a spin-off licence, a paid data feed),
both Art. 24 and Art. 34 become due **before** any commitment is made, and
Art. 15/16 put licence negotiation in ETH transfer's exclusive hands.
Revenue would then split per Art. 36: ⅓ creators, ⅓ professorship, ⅓ central ETH.

## Sign-off

| Item | Owner | Status | Date |
|---|---|---|---|
| Export-control / dual-use exemption | ETH legal / export-control office | ☐ pending | |
| IP / OSI-release authorisation | ETH Transfer | ☐ pending | |
| BROND (RU) / CENDL (CN) origin closed in writing | ETH legal | ☐ pending | |
| Art. 27(1)(a) group-leader release decision recorded | Forschungsgruppenleiter:in | ☐ pending | |
