# Changelog

## [0.11.0](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-mcp-ts-v0.16.0...nucl-parquet-mcp-ts-v0.11.0) (2026-09-29)


### ⚠ BREAKING CHANGES

* **rs:** nullable state fields in the Rust meta API (DecayEntry, EmissionEntry, CoincidenceEntry, GammaCandidate) are Option<String>, and parent_state / daughter_state no longer use the retired '' spelling; ground is 'g', undetermined is null.
* **py-mcp:** nucl-parquet-mcp now requires mcp>=2.2.0.
* **data:** rebuild all nine ENDF libraries, and bring every consumer up to the new shape ([#399](https://github.com/exoma-ch/nucl-parquet/issues/399))
* **mcp:** report the data release, not just each evaluation's version ([#385](https://github.com/exoma-ch/nucl-parquet/issues/385))

* Bump release to 0.11.0 ([#93](https://github.com/exoma-ch/nucl-parquet/issues/93)) ([17be938](https://github.com/exoma-ch/nucl-parquet/commit/17be938123238b7658ed7ae71567e85319ea45dd))


### Features

* **emissions:** Absolute per-decay photon intensities ([#196](https://github.com/exoma-ch/nucl-parquet/issues/196)) ([#197](https://github.com/exoma-ch/nucl-parquet/issues/197)) ([4e73770](https://github.com/exoma-ch/nucl-parquet/commit/4e73770865792faaaae3d64f2d6013dc57991953))
* **mcp:** Add 5 new data tools to all 3 MCP servers ([#187](https://github.com/exoma-ch/nucl-parquet/issues/187)) ([4e969c7](https://github.com/exoma-ch/nucl-parquet/commit/4e969c758ce98c5262d5cf719254efc85f0030c9)), closes [#173](https://github.com/exoma-ch/nucl-parquet/issues/173)
* **mcp:** Report the data release, not just each evaluation's version ([#385](https://github.com/exoma-ch/nucl-parquet/issues/385)) ([df66a0a](https://github.com/exoma-ch/nucl-parquet/commit/df66a0a7e893edf1ed11205a16d11e2bd46f3f91))
* **mcp:** SSoT refactor — TS + Rust MCPs use local data ([#194](https://github.com/exoma-ch/nucl-parquet/issues/194)) ([ead1297](https://github.com/exoma-ch/nucl-parquet/commit/ead129770a73eda10a12bbdf75f94a21d9ad41e8))
* **release:** Path B — per-package semver across 7 code packages (closes [#150](https://github.com/exoma-ch/nucl-parquet/issues/150)) ([#153](https://github.com/exoma-ch/nucl-parquet/issues/153)) ([1f14f52](https://github.com/exoma-ch/nucl-parquet/commit/1f14f52658949449d6fea4c11fb623d18bfd67e5))
* **rs-client:** ParquetStore — generic cached Parquet→JSON reader ([#210](https://github.com/exoma-ch/nucl-parquet/issues/210)) ([#213](https://github.com/exoma-ch/nucl-parquet/issues/213)) ([e851056](https://github.com/exoma-ch/nucl-parquet/commit/e851056382a91a74492cbc248fc3a2120a8b87b8))
* **rs:** Add DataDir auto-download + cache from GitHub Releases, closes [#31](https://github.com/exoma-ch/nucl-parquet/issues/31) ([bddff8d](https://github.com/exoma-ch/nucl-parquet/commit/bddff8d3266ccc514ce9907b41b76ad6aeb36cca))
* **rs:** Extend Rust crate with StoppingDb, CrossSectionDb, AbundancesDb, DecayDb, DoseDb ([42fded0](https://github.com/exoma-ch/nucl-parquet/commit/42fded0ff4de451d56507f241e93ee17aeaccc2b))
* **ssot:** Catalog-driven view registration + Rust MCP ParquetStore ([#206](https://github.com/exoma-ch/nucl-parquet/issues/206), [#207](https://github.com/exoma-ch/nucl-parquet/issues/207), [#217](https://github.com/exoma-ch/nucl-parquet/issues/217)) ([#218](https://github.com/exoma-ch/nucl-parquet/issues/218)) ([3fde8a4](https://github.com/exoma-ch/nucl-parquet/commit/3fde8a4a05cb7986b61dce3d2feeff97dbed9950))
* **tcs:** Materialized summing_partners table ([#177](https://github.com/exoma-ch/nucl-parquet/issues/177)) ([#195](https://github.com/exoma-ch/nucl-parquet/issues/195)) ([4d021c2](https://github.com/exoma-ch/nucl-parquet/commit/4d021c2dc2ca8e1c3a9524d90451f7e377e88bfb))


### Bug Fixes

* **data:** Rebuild all nine ENDF libraries, and bring every consumer up to the new shape ([#399](https://github.com/exoma-ch/nucl-parquet/issues/399)) ([3e16c5d](https://github.com/exoma-ch/nucl-parquet/commit/3e16c5d378d64852b2c5c8fbae25eb37b88fa3b8))
* **mcp:** Replace hardcoded version strings with package metadata ([#186](https://github.com/exoma-ch/nucl-parquet/issues/186)) ([05b352a](https://github.com/exoma-ch/nucl-parquet/commit/05b352a1443b41d944f8891d223f4545a8413c89))
* **py-mcp:** Port to mcp 2, and make the server work at all ([b9ac8b3](https://github.com/exoma-ch/nucl-parquet/commit/b9ac8b3b8f986a8f323a856f683d5a6ec48d6789))
* Repair data delivery pipeline end-to-end (closes [#35](https://github.com/exoma-ch/nucl-parquet/issues/35)) ([12eb037](https://github.com/exoma-ch/nucl-parquet/commit/12eb03788bcbfe9c38e90f16c6f7dcd88205a24a))
* Repair data delivery pipeline end-to-end (closes [#35](https://github.com/exoma-ch/nucl-parquet/issues/35)) ([47c7f43](https://github.com/exoma-ch/nucl-parquet/commit/47c7f438bbb4fa8caac0bfae34fc6b0ccd0dc1b1))
* **rs:** Extend the state vocabulary to parent_state and daughter_state ([d6985f5](https://github.com/exoma-ch/nucl-parquet/commit/d6985f5ed15bbaadc580cae36f335bb8aa2b584a))
* **ts-mcp:** Commit package-lock.json so npm ci can publish ([#166](https://github.com/exoma-ch/nucl-parquet/issues/166)) ([54e3592](https://github.com/exoma-ch/nucl-parquet/commit/54e3592389be1b687df4a8d8e67fb3be75a3d340))
* **ts-mcp:** Confine sql_query's connection to the data tree ([#424](https://github.com/exoma-ch/nucl-parquet/issues/424)) ([45adeaa](https://github.com/exoma-ch/nucl-parquet/commit/45adeaa01ae37ecdd622dc40c33236d8f6348deb))
* **ts:** Build on TypeScript 7, ship CJS types for require, catch mcp up on every major ([#412](https://github.com/exoma-ch/nucl-parquet/issues/412)) ([59ee6af](https://github.com/exoma-ch/nucl-parquet/commit/59ee6af1b3b2ba366b4bdb8ba2986a67c34c75e9))


### Refactoring

* **layout:** Move data → data/, SDKs → clients/, bump v0.3.14 ([a61afc9](https://github.com/exoma-ch/nucl-parquet/commit/a61afc918ae7c832302cbf77e7f2d3bc8597d8ba))

## [0.16.0](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-mcp-ts-v0.15.0...nucl-parquet-mcp-ts-v0.16.0) (2026-09-29)


### ⚠ BREAKING CHANGES

* **rs:** nullable state fields in the Rust meta API (DecayEntry, EmissionEntry, CoincidenceEntry, GammaCandidate) are Option<String>, and parent_state / daughter_state no longer use the retired '' spelling; ground is 'g', undetermined is null.

### Bug Fixes

* **rs:** Extend the state vocabulary to parent_state and daughter_state ([d6985f5](https://github.com/exoma-ch/nucl-parquet/commit/d6985f5ed15bbaadc580cae36f335bb8aa2b584a))

## [0.15.0](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-mcp-ts-v0.14.0...nucl-parquet-mcp-ts-v0.15.0) (2026-09-28)


### ⚠ BREAKING CHANGES

* **py-mcp:** nucl-parquet-mcp now requires mcp>=2.2.0.

### Bug Fixes

* **py-mcp:** Port to mcp 2, and make the server work at all ([b9ac8b3](https://github.com/exoma-ch/nucl-parquet/commit/b9ac8b3b8f986a8f323a856f683d5a6ec48d6789))
* **ts-mcp:** Confine sql_query's connection to the data tree ([#424](https://github.com/exoma-ch/nucl-parquet/issues/424)) ([45adeaa](https://github.com/exoma-ch/nucl-parquet/commit/45adeaa01ae37ecdd622dc40c33236d8f6348deb))
* **ts:** Build on TypeScript 7, ship CJS types for require, catch mcp up on every major ([#412](https://github.com/exoma-ch/nucl-parquet/issues/412)) ([59ee6af](https://github.com/exoma-ch/nucl-parquet/commit/59ee6af1b3b2ba366b4bdb8ba2986a67c34c75e9))

## [0.14.0](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-mcp-ts-v0.13.4...nucl-parquet-mcp-ts-v0.14.0) (2026-09-28)


### ⚠ BREAKING CHANGES

* **data:** rebuild all nine ENDF libraries, and bring every consumer up to the new shape ([#399](https://github.com/exoma-ch/nucl-parquet/issues/399))
* **mcp:** report the data release, not just each evaluation's version ([#385](https://github.com/exoma-ch/nucl-parquet/issues/385))

### Features

* **mcp:** Report the data release, not just each evaluation's version ([#385](https://github.com/exoma-ch/nucl-parquet/issues/385)) ([df66a0a](https://github.com/exoma-ch/nucl-parquet/commit/df66a0a7e893edf1ed11205a16d11e2bd46f3f91))


### Bug Fixes

* **data:** Rebuild all nine ENDF libraries, and bring every consumer up to the new shape ([#399](https://github.com/exoma-ch/nucl-parquet/issues/399)) ([3e16c5d](https://github.com/exoma-ch/nucl-parquet/commit/3e16c5d378d64852b2c5c8fbae25eb37b88fa3b8))

## [0.13.4](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-mcp-ts-v0.13.3...nucl-parquet-mcp-ts-v0.13.4) (2026-05-21)


### Features

* **emissions:** Absolute per-decay photon intensities ([#196](https://github.com/exoma-ch/nucl-parquet/issues/196)) ([#197](https://github.com/exoma-ch/nucl-parquet/issues/197)) ([4e73770](https://github.com/exoma-ch/nucl-parquet/commit/4e73770865792faaaae3d64f2d6013dc57991953))
* **mcp:** Add 5 new data tools to all 3 MCP servers ([#187](https://github.com/exoma-ch/nucl-parquet/issues/187)) ([4e969c7](https://github.com/exoma-ch/nucl-parquet/commit/4e969c758ce98c5262d5cf719254efc85f0030c9)), closes [#173](https://github.com/exoma-ch/nucl-parquet/issues/173)
* **mcp:** SSoT refactor — TS + Rust MCPs use local data ([#194](https://github.com/exoma-ch/nucl-parquet/issues/194)) ([ead1297](https://github.com/exoma-ch/nucl-parquet/commit/ead129770a73eda10a12bbdf75f94a21d9ad41e8))
* **rs-client:** ParquetStore — generic cached Parquet→JSON reader ([#210](https://github.com/exoma-ch/nucl-parquet/issues/210)) ([#213](https://github.com/exoma-ch/nucl-parquet/issues/213)) ([e851056](https://github.com/exoma-ch/nucl-parquet/commit/e851056382a91a74492cbc248fc3a2120a8b87b8))
* **ssot:** Catalog-driven view registration + Rust MCP ParquetStore ([#206](https://github.com/exoma-ch/nucl-parquet/issues/206), [#207](https://github.com/exoma-ch/nucl-parquet/issues/207), [#217](https://github.com/exoma-ch/nucl-parquet/issues/217)) ([#218](https://github.com/exoma-ch/nucl-parquet/issues/218)) ([3fde8a4](https://github.com/exoma-ch/nucl-parquet/commit/3fde8a4a05cb7986b61dce3d2feeff97dbed9950))
* **tcs:** Materialized summing_partners table ([#177](https://github.com/exoma-ch/nucl-parquet/issues/177)) ([#195](https://github.com/exoma-ch/nucl-parquet/issues/195)) ([4d021c2](https://github.com/exoma-ch/nucl-parquet/commit/4d021c2dc2ca8e1c3a9524d90451f7e377e88bfb))

## [0.13.3](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-mcp-ts-v0.13.2...nucl-parquet-mcp-ts-v0.13.3) (2026-05-15)


### Bug Fixes

* **mcp:** Replace hardcoded version strings with package metadata ([#186](https://github.com/exoma-ch/nucl-parquet/issues/186)) ([05b352a](https://github.com/exoma-ch/nucl-parquet/commit/05b352a1443b41d944f8891d223f4545a8413c89))

## [0.13.2](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-mcp-ts-v0.13.1...nucl-parquet-mcp-ts-v0.13.2) (2026-05-12)


### Bug Fixes

* **ts-mcp:** Commit package-lock.json so npm ci can publish ([#166](https://github.com/exoma-ch/nucl-parquet/issues/166)) ([54e3592](https://github.com/exoma-ch/nucl-parquet/commit/54e3592389be1b687df4a8d8e67fb3be75a3d340))

## [0.13.1](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-mcp-ts-v0.13.0...nucl-parquet-mcp-ts-v0.13.1) (2026-05-12)


### Features

* **release:** Path B — per-package semver across 7 code packages (closes [#150](https://github.com/exoma-ch/nucl-parquet/issues/150)) ([#153](https://github.com/exoma-ch/nucl-parquet/issues/153)) ([1f14f52](https://github.com/exoma-ch/nucl-parquet/commit/1f14f52658949449d6fea4c11fb623d18bfd67e5))

## Changelog

<!-- release-please prepends new release entries here. -->
<!-- Pre-per-package-semver history lives in the top-level /CHANGELOG.md. -->
