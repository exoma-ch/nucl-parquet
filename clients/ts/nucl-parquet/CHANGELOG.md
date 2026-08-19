# Changelog

## [0.16.1](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-ts-v0.16.0...nucl-parquet-ts-v0.16.1) (2026-08-19)


### Features

* **rs:** Cargo workspace, publish-race fix, and unblock the TS majors ([#311](https://github.com/exoma-ch/nucl-parquet/issues/311)) ([5799222](https://github.com/exoma-ch/nucl-parquet/commit/5799222ce3466e711029a7c14eb39b31264145b8))

## [0.16.0](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-ts-v0.15.0...nucl-parquet-ts-v0.16.0) (2026-07-10)


### ⚠ BREAKING CHANGES

* **neutron:** NJOY-processed ENDF/B-VIII.0 as a normal xs library; retire in-repo reconstruction ([#265](https://github.com/exoma-ch/nucl-parquet/issues/265))

### Features

* **neutron:** NJOY-processed ENDF/B-VIII.0 as a normal xs library; retire in-repo reconstruction ([#265](https://github.com/exoma-ch/nucl-parquet/issues/265)) ([75cd4c6](https://github.com/exoma-ch/nucl-parquet/commit/75cd4c62f13476663736e0bcd96e1d3defa3ad3a))

## [0.15.0](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-ts-v0.14.3...nucl-parquet-ts-v0.15.0) (2026-06-23)


### ⚠ BREAKING CHANGES

* **data:** federate catima heavy-ion stopping into per-isotope shards ([#252](https://github.com/exoma-ch/nucl-parquet/issues/252)) (#254)

### Features

* **data:** Federate catima heavy-ion stopping into per-isotope shards ([#252](https://github.com/exoma-ch/nucl-parquet/issues/252)) ([#254](https://github.com/exoma-ch/nucl-parquet/issues/254)) ([e9fb00f](https://github.com/exoma-ch/nucl-parquet/commit/e9fb00f3d55c0ee95e3188b96a2f9037c9e63e14))
* **parity:** Cross-language golden-file fixtures — closes [#176](https://github.com/exoma-ch/nucl-parquet/issues/176) ([#191](https://github.com/exoma-ch/nucl-parquet/issues/191)) ([179476d](https://github.com/exoma-ch/nucl-parquet/commit/179476d13d3466fd1e513563a95304a6b303a86a))
* **rs-client:** CoincidencesDb + RadiationDb with lazy loading — Sub-A of [#173](https://github.com/exoma-ch/nucl-parquet/issues/173), refs [#175](https://github.com/exoma-ch/nucl-parquet/issues/175) ([#180](https://github.com/exoma-ch/nucl-parquet/issues/180)) ([a76d52f](https://github.com/exoma-ch/nucl-parquet/commit/a76d52f1a2ca4033db478a845f6c68e369985603))


### Bug Fixes

* **ts-client:** Surface proj_A in catimaColumns (isotope resolution) ([#249](https://github.com/exoma-ch/nucl-parquet/issues/249)) ([91264be](https://github.com/exoma-ch/nucl-parquet/commit/91264be525e59883dc137bd09ac912ae68073073))

## [0.14.3](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-ts-v0.14.2...nucl-parquet-ts-v0.14.3) (2026-06-23)


### Bug Fixes

* **ts-client:** Surface proj_A in catimaColumns (isotope resolution) ([#249](https://github.com/exoma-ch/nucl-parquet/issues/249)) ([91264be](https://github.com/exoma-ch/nucl-parquet/commit/91264be525e59883dc137bd09ac912ae68073073))

## [0.14.2](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-ts-v0.14.1...nucl-parquet-ts-v0.14.2) (2026-05-21)


### Features

* **parity:** Cross-language golden-file fixtures — closes [#176](https://github.com/exoma-ch/nucl-parquet/issues/176) ([#191](https://github.com/exoma-ch/nucl-parquet/issues/191)) ([179476d](https://github.com/exoma-ch/nucl-parquet/commit/179476d13d3466fd1e513563a95304a6b303a86a))

## [0.14.1](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-ts-v0.14.0...nucl-parquet-ts-v0.14.1) (2026-05-15)


### Features

* **rs-client:** CoincidencesDb + RadiationDb with lazy loading — Sub-A of [#173](https://github.com/exoma-ch/nucl-parquet/issues/173), refs [#175](https://github.com/exoma-ch/nucl-parquet/issues/175) ([#180](https://github.com/exoma-ch/nucl-parquet/issues/180)) ([a76d52f](https://github.com/exoma-ch/nucl-parquet/commit/a76d52f1a2ca4033db478a845f6c68e369985603))

## [0.14.0](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-ts-v0.13.0...nucl-parquet-ts-v0.14.0) (2026-05-12)


### ⚠ BREAKING CHANGES

* **stopping:** route α through NIST ASTAR, ³He through catima (closes #137) ([#143](https://github.com/exoma-ch/nucl-parquet/issues/143))

### Features

* **release:** Path B — per-package semver across 7 code packages (closes [#150](https://github.com/exoma-ch/nucl-parquet/issues/150)) ([#153](https://github.com/exoma-ch/nucl-parquet/issues/153)) ([1f14f52](https://github.com/exoma-ch/nucl-parquet/commit/1f14f52658949449d6fea4c11fb623d18bfd67e5))


### Bug Fixes

* **stopping:** Route α through NIST ASTAR, ³He through catima (closes [#137](https://github.com/exoma-ch/nucl-parquet/issues/137)) ([#143](https://github.com/exoma-ch/nucl-parquet/issues/143)) ([d6beab0](https://github.com/exoma-ch/nucl-parquet/commit/d6beab000f045749b55e9ddbf7f364a8a17962ab))

## Changelog

<!-- release-please prepends new release entries here. -->
<!-- Pre-per-package-semver history lives in the top-level /CHANGELOG.md. -->
