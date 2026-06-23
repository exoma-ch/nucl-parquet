# Changelog

## [0.14.0](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-rs-v0.13.6...nucl-parquet-rs-v0.14.0) (2026-06-23)


### ⚠ BREAKING CHANGES

* **rs-client:** catima isotope resolution + repair stale Rust tests ([#247](https://github.com/exoma-ch/nucl-parquet/issues/247))

### Bug Fixes

* **rs-client:** Catima isotope resolution + repair stale Rust tests ([#247](https://github.com/exoma-ch/nucl-parquet/issues/247)) ([97be5aa](https://github.com/exoma-ch/nucl-parquet/commit/97be5aa2eed5c84c0809b26e234e87e55eb36d2f))

## [0.13.6](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-rs-v0.13.5...nucl-parquet-rs-v0.13.6) (2026-05-22)


### Features

* **rs-client:** Compound_table() + rustls TLS backend ([#226](https://github.com/exoma-ch/nucl-parquet/issues/226), [#227](https://github.com/exoma-ch/nucl-parquet/issues/227)) ([#228](https://github.com/exoma-ch/nucl-parquet/issues/228)) ([0549b8a](https://github.com/exoma-ch/nucl-parquet/commit/0549b8aa744fb4dc870cebb714f42b5bc509c381))

## [0.13.5](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-rs-v0.13.4...nucl-parquet-rs-v0.13.5) (2026-05-22)


### Features

* HTTP-backed lazy per-file fetch ([#223](https://github.com/exoma-ch/nucl-parquet/issues/223)) ([#224](https://github.com/exoma-ch/nucl-parquet/issues/224)) ([f5de182](https://github.com/exoma-ch/nucl-parquet/commit/f5de1825871e4405b682e5224ad5c39feec11e7e))

## [0.13.4](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-rs-v0.13.3...nucl-parquet-rs-v0.13.4) (2026-05-21)


### Features

* **rs-client:** Add from_bytes constructors to all typed DBs ([#221](https://github.com/exoma-ch/nucl-parquet/issues/221)) ([5691319](https://github.com/exoma-ch/nucl-parquet/commit/5691319264ba73f9152243f61bddf43c121635ba))

## [0.13.3](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-rs-v0.13.2...nucl-parquet-rs-v0.13.3) (2026-05-21)


### Features

* **parity:** Cross-language golden-file fixtures — closes [#176](https://github.com/exoma-ch/nucl-parquet/issues/176) ([#191](https://github.com/exoma-ch/nucl-parquet/issues/191)) ([179476d](https://github.com/exoma-ch/nucl-parquet/commit/179476d13d3466fd1e513563a95304a6b303a86a))
* **rs-client:** Compound stopping power + dose source attribution ([#199](https://github.com/exoma-ch/nucl-parquet/issues/199), [#200](https://github.com/exoma-ch/nucl-parquet/issues/200)) ([#201](https://github.com/exoma-ch/nucl-parquet/issues/201)) ([8a2928f](https://github.com/exoma-ch/nucl-parquet/commit/8a2928f4fdfd47d91423e45279546bf1b8130064))
* **rs-client:** ParquetStore — generic cached Parquet→JSON reader ([#210](https://github.com/exoma-ch/nucl-parquet/issues/210)) ([#213](https://github.com/exoma-ch/nucl-parquet/issues/213)) ([e851056](https://github.com/exoma-ch/nucl-parquet/commit/e851056382a91a74492cbc248fc3a2120a8b87b8))
* **rs-client:** Raw table accessors for StoppingDb + CrossSectionDb ([#202](https://github.com/exoma-ch/nucl-parquet/issues/202)) ([#205](https://github.com/exoma-ch/nucl-parquet/issues/205)) ([8727c46](https://github.com/exoma-ch/nucl-parquet/commit/8727c46bd164d0d727d4f3c35bdd5703257e3a07))

## [0.13.2](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-rs-v0.13.1...nucl-parquet-rs-v0.13.2) (2026-05-15)


### Features

* **rs-client:** CoincidencesDb + RadiationDb with lazy loading — Sub-A of [#173](https://github.com/exoma-ch/nucl-parquet/issues/173), refs [#175](https://github.com/exoma-ch/nucl-parquet/issues/175) ([#180](https://github.com/exoma-ch/nucl-parquet/issues/180)) ([a76d52f](https://github.com/exoma-ch/nucl-parquet/commit/a76d52f1a2ca4033db478a845f6c68e369985603))

## [0.13.1](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-rs-v0.13.0...nucl-parquet-rs-v0.13.1) (2026-05-12)


### Features

* **release:** Path B — per-package semver across 7 code packages (closes [#150](https://github.com/exoma-ch/nucl-parquet/issues/150)) ([#153](https://github.com/exoma-ch/nucl-parquet/issues/153)) ([1f14f52](https://github.com/exoma-ch/nucl-parquet/commit/1f14f52658949449d6fea4c11fb623d18bfd67e5))

## Changelog

<!-- release-please prepends new release entries here. -->
<!-- Pre-per-package-semver history lives in the top-level /CHANGELOG.md. -->
