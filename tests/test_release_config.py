"""Gates on the release configuration (#281).

`clients/rs/nucl-parquet-mcp` depends on its sibling by path *and* by version:

    nucl-parquet = { path = "../nucl-parquet", version = "0.15.0" }

crates.io requires the version — a path dependency without one cannot be
published, and both crates are published. So the two must move together, and
until now nothing enforced that. The comment above the line said "keep in sync"
and that was the whole mechanism.

release-please bumped `nucl-parquet` to 0.16.0 and left the constraint at
0.15.0, which does not break loudly at the config level. It breaks as:

    error: failed to select a version for the requirement `nucl-parquet = "^0.15.0"`
    candidate versions found which didn't match: 0.16.0

That is a red release PR nobody can merge, discovered only after the bot has
already opened it — and the failure names cargo resolution rather than the
config that caused it.

The config now carries an `extra-files` rule so release-please updates the
constraint alongside the version. That rule cannot be unit-tested — release
tooling is only exercised by running the bot — so this test is the detector:
if the path is wrong and the rule silently no-ops, the release PR fails *here*,
with a message that says what to fix.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent
_RS = ROOT / "clients" / "rs"


def _crate_version(crate: str) -> str:
    data = tomllib.loads((_RS / crate / "Cargo.toml").read_text())
    return data["package"]["version"]


def test_mcp_pins_the_sibling_version_it_is_built_against() -> None:
    """The declared dependency version must equal the sibling's actual version.

    Fails on any release PR where release-please bumped one and not the other,
    with a clearer diagnosis than cargo's resolution error.
    """
    actual = _crate_version("nucl-parquet")
    declared = tomllib.loads((_RS / "nucl-parquet-mcp" / "Cargo.toml").read_text())["dependencies"]["nucl-parquet"][
        "version"
    ]
    assert declared == actual, (
        f"clients/rs/nucl-parquet-mcp declares nucl-parquet = {declared!r}, but "
        f"clients/rs/nucl-parquet is version {actual!r}.\n\n"
        "cargo cannot resolve this (a 0.x bump is a breaking change), so the build "
        "fails with 'failed to select a version'. If you are looking at a "
        "release-please PR, the `extra-files` rule in release-please-config.json "
        "did not apply — check its `path`, which is resolved relative to the "
        "package directory."
    )


def test_release_please_config_has_no_parent_relative_extra_files() -> None:
    """`extra-files` paths are package-relative and may not escape upward.

    The obvious automation for the sync invariant — have release-please rewrite
    the sibling's constraint via an `extra-files` rule on
    `clients/rs/nucl-parquet` — is not expressible. release-please resolves the
    path against the package directory and rejects `..` outright:

        release-please failed: illegal pathing characters in path:
        clients/rs/nucl-parquet/../nucl-parquet-mcp/Cargo.toml

    That is not a silent no-op, which would have been survivable. It aborts the
    whole release-please run, so *no* release PR updates at all until the config
    is fixed — one bad path takes out release automation for every package in
    the repo.

    The real fix is a cargo workspace plus the `cargo-workspace` plugin, which
    is designed for exactly this and is tracked separately. Until then the
    invariant is maintained by hand and caught by
    `test_mcp_pins_the_sibling_version_it_is_built_against`.
    """
    cfg = json.loads((ROOT / "release-please-config.json").read_text())
    for name, pkg in cfg["packages"].items():
        for entry in pkg.get("extra-files", []):
            path = entry.get("path", "") if isinstance(entry, dict) else entry
            assert ".." not in path, (
                f"package {name!r} has an extra-files path containing '..' ({path!r}). "
                "release-please rejects these and aborts the entire run, so every "
                "package stops releasing — not just this one."
            )


def test_every_published_crate_has_a_release_please_entry() -> None:
    """A crate that release.yml publishes but release-please does not version
    would be published at a stale version forever."""
    cfg = json.loads((ROOT / "release-please-config.json").read_text())
    published = set(re.findall(r"clients/rs/([a-z-]+)", (ROOT / ".github" / "workflows" / "release.yml").read_text()))
    for crate in published:
        assert f"clients/rs/{crate}" in cfg["packages"], f"{crate} is published but has no release-please entry"
