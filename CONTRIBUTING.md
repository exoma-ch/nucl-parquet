# Contributing

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
