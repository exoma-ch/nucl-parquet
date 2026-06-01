# Compliance record

> **Status: DRAFT determinations — pending ETH legal / Technology Transfer sign-off.**
> Not legal advice. Tracks issues #238 (export control) and #239 (IP / release rights).

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

## 2. IP / open-source release rights (#239)

**Determination (to be confirmed by ETH Transfer):** eXoma holds the right to
release nucl-parquet's **code and the ENDF-6→Parquet conversion** under MIT, and
to redistribute the third-party data under the per-library terms in
[`ATTRIBUTION.md`](ATTRIBUTION.md). Copyright holder string: **"eXoma (Exotic
Matter Applications), ETH Zürich."**

## Sign-off

| Item | Owner | Status | Date |
|---|---|---|---|
| Export-control / dual-use exemption | ETH legal / export-control office | ☐ pending | |
| IP / OSI-release authorisation | ETH Transfer | ☐ pending | |
| BROND (RU) / CENDL (CN) origin closed in writing | ETH legal | ☐ pending | |
