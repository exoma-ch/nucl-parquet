# Contributing

## Contribution license (inbound = outbound)

By submitting a contribution you certify that you wrote it (or have the right to
submit it) and you agree it is licensed under the project's terms — **MIT** for
code and the ENDF-6→Parquet conversion. Do not contribute third-party nuclear
data without recording its provenance and terms in [`data/licenses.toml`](data/licenses.toml)
and [`ATTRIBUTION.md`](ATTRIBUTION.md) (see also [`NOTICE`](NOTICE)). We follow the
[Developer Certificate of Origin](https://developercertificate.org/); sign off
your commits with `git commit -s`.

## Pre-commit hooks

This repo uses [prek](https://github.com/j178/prek) (a Rust-native `pre-commit` reimplementation) for local lint/format/clippy gates. CI enforces the same checks via `prek run --all-files`.

### Install

```bash
brew install prek           # macOS
# or
cargo install --locked prek # any platform with cargo
# or
uv tool install prek        # already required for the Python dev setup
```

### Enable hooks in your clone

```bash
prek install                              # commit-time hooks (ruff, cargo fmt, trivia)
prek install --hook-type pre-push         # push-time hooks (cargo clippy)
```

### Run manually

```bash
prek run --all-files                      # commit-stage hooks
prek run --all-files --hook-stage pre-push  # clippy
```

Commit-stage hooks are cheap; pre-push runs `cargo clippy -- -D warnings` on both `clients/rs/nucl-parquet` and `clients/rs/nucl-parquet-mcp`.

If you must bypass, use `git commit --no-verify` or `git push --no-verify` — CI will still catch you.

## Conventional Commits and per-package releases

This repo runs [release-please](https://github.com/googleapis/release-please) in **per-package** mode (see `release-please-config.json`). Each code package has its own semver track, its own tag prefix, and its own CHANGELOG. The commit's *file paths* decide which package(s) get bumped; the commit's *scope* is for human readability and changelog filtering.

### Package map

| Package | Path | Release-please component | Tag prefix |
|---|---|---|---|
| `nucl-parquet` (Python) | `.` | `nucl-parquet-py` | `nucl-parquet-py-v` |
| `nucl-parquet-mcp` (Python) | `clients/py/nucl-parquet-mcp/` | `nucl-parquet-mcp-py` | `nucl-parquet-mcp-py-v` |
| `nucl-parquet` (Rust) | `clients/rs/nucl-parquet/` | `nucl-parquet-rs` | `nucl-parquet-rs-v` |
| `nucl-parquet-mcp` (Rust) | `clients/rs/nucl-parquet-mcp/` | `nucl-parquet-mcp-rs` | `nucl-parquet-mcp-rs-v` |
| `@nucl-parquet/core` (TS) | `clients/ts/nucl-parquet/` | `nucl-parquet-ts` | `nucl-parquet-ts-v` |
| `@nucl-parquet/mcp` (TS) | `clients/ts/nucl-parquet-mcp/` | `nucl-parquet-mcp-ts` | `nucl-parquet-mcp-ts-v` |
| `nucl-parquet-go` | `clients/go/nucl-parquet/` | `clients/go/nucl-parquet` | `clients/go/nucl-parquet/v` |

### Recommended commit scopes

Scopes are conventional, not strictly enforced — but consistent scopes make the changelog readable:

| Scope | Meaning | Likely bumps |
|---|---|---|
| `py`, `py-core` | Python core (loader, builders, tests) | `nucl-parquet-py` |
| `py-mcp` | Python MCP server | `nucl-parquet-mcp-py` |
| `rs`, `rs-core` | Rust core crate | `nucl-parquet-rs` |
| `rs-mcp` | Rust MCP server | `nucl-parquet-mcp-rs` |
| `ts`, `ts-core` | TypeScript core | `nucl-parquet-ts` |
| `ts-mcp` | TypeScript MCP server | `nucl-parquet-mcp-ts` |
| `go` | Go module | `clients/go/nucl-parquet` |
| domain (e.g. `stopping`, `em`, `nudex`) | Cross-cutting nuclear-data work that touches the Python core + tests + data | usually `nucl-parquet-py` (and any client whose tests/docs were updated) |
| `release`, `ci`, `data` | Build/release plumbing | usually no bump (chore-equivalent) |

The path-detection logic is authoritative: if your commit touches files inside `clients/py/nucl-parquet-mcp/`, that package bumps regardless of scope.

### Examples

```text
fix(stopping)!: route α through NIST ASTAR
# Touches nucl_parquet/, tests/, data/, README.md.
# → Bumps nucl-parquet-py only.

chore(rs-mcp): drop ASTAR refs from tool schema
# Touches clients/rs/nucl-parquet-mcp/src/main.rs.
# → Bumps nucl-parquet-mcp-rs only.

docs(release): document the dual-track scheme
# Touches README.md only.
# → Bumps nucl-parquet-py only (root-owned README).

feat(em): add Seltzer-Berger bremsstrahlung DCS
# Touches nucl_parquet/, tests/, data/.
# → Bumps nucl-parquet-py only.
```

### Cross-package changes

If a single change has to touch multiple packages (e.g. you add a new column to the parquet schema and update every client), prefer **separate commits per scope** so each package's CHANGELOG describes the change in package-relevant language. Squash-merge into `main` keeps the topology clean.

### Breaking changes

A `!` after the scope (or a `BREAKING CHANGE:` footer) signals a major bump. In a 0.x repo this manifests as a minor bump per release-please's defaults. Use it.

### Data releases

Data lives outside the per-package code semver, on its own CalVer tag namespace. Conventional-commit scopes for data work (`feat(data)`, etc.) do not trigger any code-package bump.

Releasing data is one edit: bump `data/catalog.json::data_version` to the next `YYYY.MM.MICRO` and merge to `main`. Do not push a tag by hand. `.github/workflows/auto-tag-data.yml` pushes `data-<version>` with the release-bot App token on merge, and that tag push fires `.github/workflows/release-data.yml`, which re-validates tag-vs-catalog and publishes the signed tarball, manifest and signatures.

If a version is tagged but has no release — the #344 failure — re-run **Auto-tag data release** from the Actions tab; it reconciles the catalog against what is actually published and re-triggers the release. The equivalent by hand is:

```console
$ gh workflow run release-data.yml -f tag=data-YYYY.MM.MICRO
```
