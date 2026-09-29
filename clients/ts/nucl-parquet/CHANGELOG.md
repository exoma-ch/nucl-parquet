# Changelog

## [0.11.0](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-ts-v0.18.0...nucl-parquet-ts-v0.11.0) (2026-09-29)


### ⚠ BREAKING CHANGES

* **rs:** nullable state fields in the Rust meta API (DecayEntry, EmissionEntry, CoincidenceEntry, GammaCandidate) are Option<String>, and parent_state / daughter_state no longer use the retired '' spelling; ground is 'g', undetermined is null.
* **data:** rebuild all nine ENDF libraries, and bring every consumer up to the new shape ([#399](https://github.com/exoma-ch/nucl-parquet/issues/399))
* **clients:** stop reading a null residual_Z or state as a real value ([#381](https://github.com/exoma-ch/nucl-parquet/issues/381))
* **neutron:** NJOY-processed ENDF/B-VIII.0 as a normal xs library; retire in-repo reconstruction ([#265](https://github.com/exoma-ch/nucl-parquet/issues/265))
* **data:** federate catima heavy-ion stopping into per-isotope shards ([#252](https://github.com/exoma-ch/nucl-parquet/issues/252)) (#254)
* **stopping:** route α through NIST ASTAR, ³He through catima (closes #137) ([#143](https://github.com/exoma-ch/nucl-parquet/issues/143))

* Bump release to 0.11.0 ([#93](https://github.com/exoma-ch/nucl-parquet/issues/93)) ([17be938](https://github.com/exoma-ch/nucl-parquet/commit/17be938123238b7658ed7ae71567e85319ea45dd))


### Features

* **data:** Federate catima heavy-ion stopping into per-isotope shards ([#252](https://github.com/exoma-ch/nucl-parquet/issues/252)) ([#254](https://github.com/exoma-ch/nucl-parquet/issues/254)) ([e9fb00f](https://github.com/exoma-ch/nucl-parquet/commit/e9fb00f3d55c0ee95e3188b96a2f9037c9e63e14))
* **neutron:** NJOY-processed ENDF/B-VIII.0 as a normal xs library; retire in-repo reconstruction ([#265](https://github.com/exoma-ch/nucl-parquet/issues/265)) ([75cd4c6](https://github.com/exoma-ch/nucl-parquet/commit/75cd4c62f13476663736e0bcd96e1d3defa3ad3a))
* **parity:** Cross-language golden-file fixtures — closes [#176](https://github.com/exoma-ch/nucl-parquet/issues/176) ([#191](https://github.com/exoma-ch/nucl-parquet/issues/191)) ([179476d](https://github.com/exoma-ch/nucl-parquet/commit/179476d13d3466fd1e513563a95304a6b303a86a))
* **release:** Path B — per-package semver across 7 code packages (closes [#150](https://github.com/exoma-ch/nucl-parquet/issues/150)) ([#153](https://github.com/exoma-ch/nucl-parquet/issues/153)) ([1f14f52](https://github.com/exoma-ch/nucl-parquet/commit/1f14f52658949449d6fea4c11fb623d18bfd67e5))
* **rs-client:** CoincidencesDb + RadiationDb with lazy loading — Sub-A of [#173](https://github.com/exoma-ch/nucl-parquet/issues/173), refs [#175](https://github.com/exoma-ch/nucl-parquet/issues/175) ([#180](https://github.com/exoma-ch/nucl-parquet/issues/180)) ([a76d52f](https://github.com/exoma-ch/nucl-parquet/commit/a76d52f1a2ca4033db478a845f6c68e369985603))
* **rs:** Add DataDir auto-download + cache from GitHub Releases, closes [#31](https://github.com/exoma-ch/nucl-parquet/issues/31) ([bddff8d](https://github.com/exoma-ch/nucl-parquet/commit/bddff8d3266ccc514ce9907b41b76ad6aeb36cca))
* **rs:** Cargo workspace, publish-race fix, and unblock the TS majors ([#311](https://github.com/exoma-ch/nucl-parquet/issues/311)) ([5799222](https://github.com/exoma-ch/nucl-parquet/commit/5799222ce3466e711029a7c14eb39b31264145b8))
* **rs:** Extend Rust crate with StoppingDb, CrossSectionDb, AbundancesDb, DecayDb, DoseDb ([42fded0](https://github.com/exoma-ch/nucl-parquet/commit/42fded0ff4de451d56507f241e93ee17aeaccc2b))
* **stopping:** Add energy straggling column to catima tables ([742694e](https://github.com/exoma-ch/nucl-parquet/commit/742694e6e8eddafb0c2b4634757a38b7523e18bf))
* **stopping:** Add energy straggling column to catima tables, closes [#25](https://github.com/exoma-ch/nucl-parquet/issues/25) ([bf75828](https://github.com/exoma-ch/nucl-parquet/commit/bf7582865e6ab66280adf5008a05f5639ff03a8e))
* **ts:** Add stoppingColumns/xsColumns for zero-copy WASM transfer, closes [#23](https://github.com/exoma-ch/nucl-parquet/issues/23) ([1e78c5b](https://github.com/exoma-ch/nucl-parquet/commit/1e78c5baf0a230321dccb187a1021641e2e8ff4e))


### Bug Fixes

* **clients:** Stop reading a null residual_Z or state as a real value ([#381](https://github.com/exoma-ch/nucl-parquet/issues/381)) ([7fc83c5](https://github.com/exoma-ch/nucl-parquet/commit/7fc83c5f7bef07c7fdc1371c085927a4e81de943))
* **data:** Rebuild all nine ENDF libraries, and bring every consumer up to the new shape ([#399](https://github.com/exoma-ch/nucl-parquet/issues/399)) ([3e16c5d](https://github.com/exoma-ch/nucl-parquet/commit/3e16c5d378d64852b2c5c8fbae25eb37b88fa3b8))
* **rs:** Extend the state vocabulary to parent_state and daughter_state ([d6985f5](https://github.com/exoma-ch/nucl-parquet/commit/d6985f5ed15bbaadc580cae36f335bb8aa2b584a))
* **stopping:** Route α through NIST ASTAR, ³He through catima (closes [#137](https://github.com/exoma-ch/nucl-parquet/issues/137)) ([#143](https://github.com/exoma-ch/nucl-parquet/issues/143)) ([d6beab0](https://github.com/exoma-ch/nucl-parquet/commit/d6beab000f045749b55e9ddbf7f364a8a17962ab))
* **ts-client:** Surface proj_A in catimaColumns (isotope resolution) ([#249](https://github.com/exoma-ch/nucl-parquet/issues/249)) ([91264be](https://github.com/exoma-ch/nucl-parquet/commit/91264be525e59883dc137bd09ac912ae68073073))
* **ts:** Build @nucl-parquet/core with TypeScript 5, and build it in CI ([#409](https://github.com/exoma-ch/nucl-parquet/issues/409)) ([ae89e11](https://github.com/exoma-ch/nucl-parquet/commit/ae89e116cfaa0812dabd0a8c2633654445cdf851))
* **ts:** Build on TypeScript 7, ship CJS types for require, catch mcp up on every major ([#412](https://github.com/exoma-ch/nucl-parquet/issues/412)) ([59ee6af](https://github.com/exoma-ch/nucl-parquet/commit/59ee6af1b3b2ba366b4bdb8ba2986a67c34c75e9))


### Refactoring

* **layout:** Move data → data/, SDKs → clients/, bump v0.3.14 ([a61afc9](https://github.com/exoma-ch/nucl-parquet/commit/a61afc918ae7c832302cbf77e7f2d3bc8597d8ba))

## [0.18.0](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-ts-v0.17.2...nucl-parquet-ts-v0.18.0) (2026-09-29)


### ⚠ BREAKING CHANGES

* **rs:** nullable state fields in the Rust meta API (DecayEntry, EmissionEntry, CoincidenceEntry, GammaCandidate) are Option<String>, and parent_state / daughter_state no longer use the retired '' spelling; ground is 'g', undetermined is null.

### Bug Fixes

* **rs:** Extend the state vocabulary to parent_state and daughter_state ([d6985f5](https://github.com/exoma-ch/nucl-parquet/commit/d6985f5ed15bbaadc580cae36f335bb8aa2b584a))

## [0.17.2](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-ts-v0.17.1...nucl-parquet-ts-v0.17.2) (2026-09-28)


### Bug Fixes

* **ts:** Build on TypeScript 7, ship CJS types for require, catch mcp up on every major ([#412](https://github.com/exoma-ch/nucl-parquet/issues/412)) ([59ee6af](https://github.com/exoma-ch/nucl-parquet/commit/59ee6af1b3b2ba366b4bdb8ba2986a67c34c75e9))

## [0.17.1](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-ts-v0.17.0...nucl-parquet-ts-v0.17.1) (2026-09-28)


### Bug Fixes

* **ts:** Build @nucl-parquet/core with TypeScript 5, and build it in CI ([#409](https://github.com/exoma-ch/nucl-parquet/issues/409)) ([ae89e11](https://github.com/exoma-ch/nucl-parquet/commit/ae89e116cfaa0812dabd0a8c2633654445cdf851))

## [0.17.0](https://github.com/exoma-ch/nucl-parquet/compare/nucl-parquet-ts-v0.16.1...nucl-parquet-ts-v0.17.0) (2026-09-28)


### ⚠ BREAKING CHANGES

* **data:** rebuild all nine ENDF libraries, and bring every consumer up to the new shape ([#399](https://github.com/exoma-ch/nucl-parquet/issues/399))
* **clients:** stop reading a null residual_Z or state as a real value ([#381](https://github.com/exoma-ch/nucl-parquet/issues/381))

### Bug Fixes

* **clients:** Stop reading a null residual_Z or state as a real value ([#381](https://github.com/exoma-ch/nucl-parquet/issues/381)) ([7fc83c5](https://github.com/exoma-ch/nucl-parquet/commit/7fc83c5f7bef07c7fdc1371c085927a4e81de943))
* **data:** Rebuild all nine ENDF libraries, and bring every consumer up to the new shape ([#399](https://github.com/exoma-ch/nucl-parquet/issues/399)) ([3e16c5d](https://github.com/exoma-ch/nucl-parquet/commit/3e16c5d378d64852b2c5c8fbae25eb37b88fa3b8))

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
