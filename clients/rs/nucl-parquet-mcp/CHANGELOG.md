# Changelog

## [0.13.4](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-mcp-rs-v0.13.3...nucl-parquet-mcp-rs-v0.13.4) (2026-05-21)


### Features

* **emissions:** Absolute per-decay photon intensities ([#196](https://github.com/exoma-ch/nucl-parquet/issues/196)) ([#197](https://github.com/exoma-ch/nucl-parquet/issues/197)) ([4e73770](https://github.com/exoma-ch/nucl-parquet/commit/4e73770865792faaaae3d64f2d6013dc57991953))
* **mcp:** Add 5 new data tools to all 3 MCP servers ([#187](https://github.com/exoma-ch/nucl-parquet/issues/187)) ([4e969c7](https://github.com/exoma-ch/nucl-parquet/commit/4e969c758ce98c5262d5cf719254efc85f0030c9)), closes [#173](https://github.com/exoma-ch/nucl-parquet/issues/173)
* **mcp:** SSoT refactor — TS + Rust MCPs use local data ([#194](https://github.com/exoma-ch/nucl-parquet/issues/194)) ([ead1297](https://github.com/exoma-ch/nucl-parquet/commit/ead129770a73eda10a12bbdf75f94a21d9ad41e8))
* **rs-client:** ParquetStore — generic cached Parquet→JSON reader ([#210](https://github.com/exoma-ch/nucl-parquet/issues/210)) ([#213](https://github.com/exoma-ch/nucl-parquet/issues/213)) ([e851056](https://github.com/exoma-ch/nucl-parquet/commit/e851056382a91a74492cbc248fc3a2120a8b87b8))
* **ssot:** Catalog-driven view registration + Rust MCP ParquetStore ([#206](https://github.com/exoma-ch/nucl-parquet/issues/206), [#207](https://github.com/exoma-ch/nucl-parquet/issues/207), [#217](https://github.com/exoma-ch/nucl-parquet/issues/217)) ([#218](https://github.com/exoma-ch/nucl-parquet/issues/218)) ([3fde8a4](https://github.com/exoma-ch/nucl-parquet/commit/3fde8a4a05cb7986b61dce3d2feeff97dbed9950))
* **tcs:** Materialized summing_partners table ([#177](https://github.com/exoma-ch/nucl-parquet/issues/177)) ([#195](https://github.com/exoma-ch/nucl-parquet/issues/195)) ([4d021c2](https://github.com/exoma-ch/nucl-parquet/commit/4d021c2dc2ca8e1c3a9524d90451f7e377e88bfb))

## [0.13.3](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-mcp-rs-v0.13.2...nucl-parquet-mcp-rs-v0.13.3) (2026-05-15)


### Bug Fixes

* **mcp:** Replace hardcoded version strings with package metadata ([#186](https://github.com/exoma-ch/nucl-parquet/issues/186)) ([05b352a](https://github.com/exoma-ch/nucl-parquet/commit/05b352a1443b41d944f8891d223f4545a8413c89))

## [0.13.2](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-mcp-rs-v0.13.1...nucl-parquet-mcp-rs-v0.13.2) (2026-05-12)


### Bug Fixes

* **rs-mcp:** Enable parquet zstd feature so server reads live data ([#164](https://github.com/exoma-ch/nucl-parquet/issues/164)) ([6e7046d](https://github.com/exoma-ch/nucl-parquet/commit/6e7046dea04ad58c2a3168ce4620d691b076e56e))

## [0.13.1](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-mcp-rs-v0.13.0...nucl-parquet-mcp-rs-v0.13.1) (2026-05-12)


### Features

* **release:** Path B — per-package semver across 7 code packages (closes [#150](https://github.com/exoma-ch/nucl-parquet/issues/150)) ([#153](https://github.com/exoma-ch/nucl-parquet/issues/153)) ([1f14f52](https://github.com/exoma-ch/nucl-parquet/commit/1f14f52658949449d6fea4c11fb623d18bfd67e5))

## Changelog

<!-- release-please prepends new release entries here. -->
<!-- Pre-per-package-semver history lives in the top-level /CHANGELOG.md. -->
