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

`clients/rs` is now a real cargo workspace, so release-please's
`cargo-workspace` plugin maintains the constraint. That plugin cannot be
unit-tested — release tooling is only exercised by running the bot — so these
tests are the detector: if it silently stops working, the release PR fails
*here*, naming both versions, rather than as a cargo resolution error.

Reads `Cargo.toml` and the release config from the checkout — no data tree, no
network.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
_RS = ROOT / "clients" / "rs"


def _crate_version(crate: str) -> str:
    data = tomllib.loads((_RS / crate / "Cargo.toml").read_text())
    return data["package"]["version"]


def _caret_compatible(declared: str, actual: str) -> bool:
    """Does cargo's default (caret) requirement `declared` admit `actual`?

    For 0.x, caret pins the *minor*: `^0.15.0` admits 0.15.9 but not 0.16.0.
    From 1.0 it pins the major. This is the rule that decides whether a release
    PR can resolve, so it is the rule the test should encode.
    """
    d = [int(x) for x in declared.split(".")]
    a = [int(x) for x in actual.split(".")]
    if a < d:
        return False
    return (d[0], a[0]) == (0, 0) and d[1] == a[1] if d[0] == 0 else d[0] == a[0]


@pytest.mark.parametrize(
    ("declared", "actual", "admits"),
    [
        # The #281 failure: a 0.x minor bump is breaking, so the old pin cannot resolve.
        ("0.15.0", "0.16.0", False),
        # A patch bump of the core crate must NOT require a manual edit.
        ("0.16.0", "0.16.1", True),
        ("0.16.0", "0.16.0", True),
        # Declaring a version newer than what exists is unresolvable.
        ("0.16.2", "0.16.1", False),
        # From 1.0 caret pins the major instead of the minor.
        ("1.2.0", "1.5.0", True),
        ("1.2.0", "2.0.0", False),
    ],
)
def test_caret_rule_matches_cargo(declared: str, actual: str, admits: bool) -> None:
    """The helper must encode cargo's rule, not an approximation of it.

    Getting this wrong in either direction is costly: too strict blocks valid
    patch releases, too loose lets an unresolvable release PR through to a
    confusing cargo error.
    """
    assert _caret_compatible(declared, actual) is admits


def test_mcp_pins_a_version_cargo_can_resolve() -> None:
    """The declared dependency must admit the sibling's actual version.

    This is the check that #281 needed: release-please bumped nucl-parquet
    0.15.0 -> 0.16.0 and left the constraint at 0.15.0, so cargo could not
    resolve and the release PR was unmergeable.

    Deliberately semver-compatibility rather than string equality. Exact
    equality is a stronger claim than cargo requires, and it would fail every
    *patch* release of the core crate — 0.16.1 against a `0.16.0` pin resolves
    perfectly well. An earlier version of this test asserted equality and blocked
    exactly that, which is a test inventing a constraint rather than gating one.
    """
    actual = _crate_version("nucl-parquet")
    declared = tomllib.loads((_RS / "nucl-parquet-mcp" / "Cargo.toml").read_text())["dependencies"]["nucl-parquet"][
        "version"
    ]
    assert _caret_compatible(declared, actual), (
        f"clients/rs/nucl-parquet-mcp declares nucl-parquet = {declared!r}, which does "
        f"not admit the sibling's actual version {actual!r}.\n\n"
        "cargo resolves the caret requirement against the registry, so this fails as "
        "'failed to select a version'. On a release-please PR, edit the constraint in "
        "clients/rs/nucl-parquet-mcp/Cargo.toml — the documented one-line step, since "
        "neither extra-files nor the cargo-workspace plugin can do it here (#307, #318)."
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


# -- The workspace (#307, #309) ---------------------------------------------


def test_rust_clients_are_one_cargo_workspace() -> None:
    """Both crates must be members of a real workspace.

    Two crates in sibling directories are not a workspace, and the difference
    is not cosmetic: release-please's `cargo-workspace` plugin only maintains
    intra-workspace dependency versions for actual members. Before this, the
    sibling pin was maintained by a comment.
    """
    ws = tomllib.loads((_RS / "Cargo.toml").read_text())
    assert set(ws["workspace"]["members"]) == {"nucl-parquet", "nucl-parquet-mcp"}


def test_one_lockfile_for_the_workspace() -> None:
    """A workspace resolves once, so there is exactly one tracked lockfile.

    Previously `nucl-parquet/Cargo.lock` was committed and the mcp one was
    gitignored — one published binary had a reproducible dependency set and the
    other did not.
    """
    assert (_RS / "Cargo.lock").exists(), "the workspace must have a lockfile"
    for crate in ("nucl-parquet", "nucl-parquet-mcp"):
        assert not (_RS / crate / "Cargo.lock").exists(), f"{crate} still has its own lockfile"


def test_shared_arrow_parquet_versions_are_inherited() -> None:
    """arrow and parquet ship from one upstream workspace on a lockstep version.

    Declaring them per-crate makes a mismatch representable, and a mismatch is
    always a bug. Inheriting makes it unrepresentable and gives Dependabot one
    line to bump instead of two.
    """
    ws = tomllib.loads((_RS / "Cargo.toml").read_text())["workspace"]["dependencies"]
    assert ws["arrow"]["version"] == ws["parquet"]["version"]
    core = tomllib.loads((_RS / "nucl-parquet" / "Cargo.toml").read_text())
    for dep in ("arrow", "parquet"):
        assert core["dependencies"][dep].get("workspace") is True, f"{dep} should inherit from the workspace"


def test_mcp_publish_waits_for_its_dependency() -> None:
    """The dependent crate must not race its dependency onto crates.io (#309).

    Both rust publishes are tag-triggered and release-please tags everything at
    once, so they run concurrently with no ordering. `cargo publish` resolves
    the sibling against the registry, so publishing first fails — about half the
    time, self-healing on re-run, which is how it reads as flake.
    """
    import yaml

    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "release.yml").read_text())
    job = wf["jobs"]["cargo-mcp"]
    names = [s.get("name", "") for s in job["steps"]]
    assert any("Wait for nucl-parquet" in n for n in names), "the mcp publish must wait for its dependency"
    wait = next(s for s in job["steps"] if "Wait for nucl-parquet" in s.get("name", ""))
    assert "crates.io/api/v1/crates/nucl-parquet/" in wait["run"], "it must poll the registry, not just sleep"
    idx_wait = names.index(next(n for n in names if "Wait for nucl-parquet" in n))
    idx_pub = next(i for i, s in enumerate(job["steps"]) if "cargo publish" in str(s.get("run", "")))
    assert idx_wait < idx_pub, "the wait must precede the publish"


def test_cargo_workspace_plugin_is_not_enabled() -> None:
    """The plugin cannot be used here: it assumes the repo root is the workspace.

    Enabling it looked right — it is the tool built for exactly this invariant —
    but it resolves the workspace manifest at the repository root, and this
    workspace is nested at `clients/rs/`:

        running plugin: CargoWorkspace
        Fetching Cargo.toml from branch main
        release-please failed: Failed to find file: Cargo.toml

    As with the `extra-files` attempt before it, the failure is not local to the
    rust packages — it aborts the entire release-please run, so no package
    releases until the config is fixed. Two different attempts to automate this
    invariant have now taken out release automation for the whole repo, which is
    worth more than the manual edit they were trying to remove.

    The sibling pin therefore stays a documented one-line edit at release time,
    caught by `test_mcp_pins_the_sibling_version_it_is_built_against`. The
    workspace itself is still worth having — one lockfile, `--all-targets`
    linting, shared dependency versions — none of which depended on the plugin.
    """
    cfg = json.loads((ROOT / "release-please-config.json").read_text())
    assert "cargo-workspace" not in cfg["plugins"], (
        "the cargo-workspace plugin resolves Cargo.toml at the repo root and this "
        "workspace is at clients/rs/ — enabling it aborts every release-please run"
    )
