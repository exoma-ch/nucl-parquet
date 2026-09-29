# Changelog

## [0.15.0](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-mcp-py-v0.14.0...nucl-parquet-mcp-py-v0.15.0) (2026-09-29)


### ⚠ BREAKING CHANGES

* **rs:** nullable state fields in the Rust meta API (DecayEntry, EmissionEntry, CoincidenceEntry, GammaCandidate) are Option<String>, and parent_state / daughter_state no longer use the retired '' spelling; ground is 'g', undetermined is null.

### Bug Fixes

* **rs:** Extend the state vocabulary to parent_state and daughter_state ([d6985f5](https://github.com/exoma-ch/nucl-parquet/commit/d6985f5ed15bbaadc580cae36f335bb8aa2b584a))

## [0.14.0](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-mcp-py-v0.13.3...nucl-parquet-mcp-py-v0.14.0) (2026-09-28)


### ⚠ BREAKING CHANGES

* **py-mcp:** nucl-parquet-mcp now requires mcp>=2.2.0.

### Bug Fixes

* **py-mcp:** Port to mcp 2, and make the server work at all ([b9ac8b3](https://github.com/exoma-ch/nucl-parquet/commit/b9ac8b3b8f986a8f323a856f683d5a6ec48d6789))

## [0.13.3](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-mcp-py-v0.13.2...nucl-parquet-mcp-py-v0.13.3) (2026-05-21)


### Features

* **emissions:** Absolute per-decay photon intensities ([#196](https://github.com/exoma-ch/nucl-parquet/issues/196)) ([#197](https://github.com/exoma-ch/nucl-parquet/issues/197)) ([4e73770](https://github.com/exoma-ch/nucl-parquet/commit/4e73770865792faaaae3d64f2d6013dc57991953))
* **mcp:** Add 5 new data tools to all 3 MCP servers ([#187](https://github.com/exoma-ch/nucl-parquet/issues/187)) ([4e969c7](https://github.com/exoma-ch/nucl-parquet/commit/4e969c758ce98c5262d5cf719254efc85f0030c9)), closes [#173](https://github.com/exoma-ch/nucl-parquet/issues/173)
* **mcp:** SSoT refactor — TS + Rust MCPs use local data ([#194](https://github.com/exoma-ch/nucl-parquet/issues/194)) ([ead1297](https://github.com/exoma-ch/nucl-parquet/commit/ead129770a73eda10a12bbdf75f94a21d9ad41e8))
* **py-mcp:** SSoT refactor — wrap nucl_parquet client library ([#192](https://github.com/exoma-ch/nucl-parquet/issues/192)) ([1867306](https://github.com/exoma-ch/nucl-parquet/commit/1867306d33b353118e410aa0ff531718325547dc))
* **rs-client:** ParquetStore — generic cached Parquet→JSON reader ([#210](https://github.com/exoma-ch/nucl-parquet/issues/210)) ([#213](https://github.com/exoma-ch/nucl-parquet/issues/213)) ([e851056](https://github.com/exoma-ch/nucl-parquet/commit/e851056382a91a74492cbc248fc3a2120a8b87b8))
* **tcs:** Materialized summing_partners table ([#177](https://github.com/exoma-ch/nucl-parquet/issues/177)) ([#195](https://github.com/exoma-ch/nucl-parquet/issues/195)) ([4d021c2](https://github.com/exoma-ch/nucl-parquet/commit/4d021c2dc2ca8e1c3a9524d90451f7e377e88bfb))

## [0.13.2](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-mcp-py-v0.13.1...nucl-parquet-mcp-py-v0.13.2) (2026-05-15)


### Bug Fixes

* **mcp:** Replace hardcoded version strings with package metadata ([#186](https://github.com/exoma-ch/nucl-parquet/issues/186)) ([05b352a](https://github.com/exoma-ch/nucl-parquet/commit/05b352a1443b41d944f8891d223f4545a8413c89))

## [0.13.1](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-mcp-py-v0.13.0...nucl-parquet-mcp-py-v0.13.1) (2026-05-12)


### Features

* **release:** Path B — per-package semver across 7 code packages (closes [#150](https://github.com/exoma-ch/nucl-parquet/issues/150)) ([#153](https://github.com/exoma-ch/nucl-parquet/issues/153)) ([1f14f52](https://github.com/exoma-ch/nucl-parquet/commit/1f14f52658949449d6fea4c11fb623d18bfd67e5))

## Changelog

<!-- release-please prepends new release entries here. -->
<!-- Pre-per-package-semver history lives in the top-level /CHANGELOG.md. -->
