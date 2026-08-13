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


def test_release_please_is_configured_to_keep_them_in_sync() -> None:
    """The automation must exist, not just the check.

    Without the rule this invariant is maintained by a comment, which is how it
    broke: every release PR would need a manual edit before it could go green.
    """
    cfg = json.loads((ROOT / "release-please-config.json").read_text())
    extra = cfg["packages"]["clients/rs/nucl-parquet"].get("extra-files", [])
    targets = [e for e in extra if isinstance(e, dict) and "nucl-parquet-mcp/Cargo.toml" in e.get("path", "")]
    assert targets, (
        "release-please must update the sibling's dependency constraint when it "
        "bumps clients/rs/nucl-parquet, or every release PR opens red."
    )
    rule = targets[0]
    assert rule.get("type") == "toml"
    assert rule.get("jsonpath") == "$.dependencies.nucl-parquet.version"


def test_every_published_crate_has_a_release_please_entry() -> None:
    """A crate that release.yml publishes but release-please does not version
    would be published at a stale version forever."""
    cfg = json.loads((ROOT / "release-please-config.json").read_text())
    published = set(re.findall(r"clients/rs/([a-z-]+)", (ROOT / ".github" / "workflows" / "release.yml").read_text()))
    for crate in published:
        assert f"clients/rs/{crate}" in cfg["packages"], f"{crate} is published but has no release-please entry"
