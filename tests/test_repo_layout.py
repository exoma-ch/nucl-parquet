"""Gates on where data is allowed to live in the checkout (#341).

`hi-xs-prod/` (552 files) and `tendl-2025/` (71) sat at the *repository root*,
shadowing `data/hi-xs-prod/` and `data/tendl-2025/`. Nothing read them —
`catalog.json` paths are relative to `data/`, and the release tarball is built
`tar --zstd -C data`, so a root-level tree never enters it. They were the
pre-migration 6-column form of files that `data/` already carried in canonical
18-column form, kept alive only by the tooling that created them.

Two scripts wrote them, and both defaulted `--output` to the repo root while
writing `<output>/<subdir>/`:

    scripts/fetch_endf_libs.py  ->  <root>/<library>/xs/
    scripts/fetch_exfor.py      ->  <root>/exfor/

So the documented ingest command put fresh parquets into a tracked top-level
directory, mixing untracked output with committed files. The #334 re-ingest hit
exactly this and had to be undone with `git checkout` + `git clean`.

The two tests below close the two halves. `test_no_library_shaped_directory_at_repo_root`
catches the artefact; the `--output` default tests catch the cause.

Deliberately *not* a `.gitignore` entry for `hi-xs-prod/` and `tendl-2025/`. An
ignore rule makes a re-scattered ingest invisible — the files land next to `data/`
and nothing says so — which is strictly worse than the loud failure here. It would
also mask this very test if the test scanned the filesystem. A stale tree should
fail the suite by name, not disappear from `git status`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from _paths import DATA_DIR  # noqa: E402

# A directory holding either of these is carrying shipped data, and shipped data
# belongs under `data/`. `manifest.json` is matched on the exact basename so the
# repo-root `.release-please-manifest.json` is not a false positive.
_DATA_MARKERS = (".parquet",)
_DATA_MARKER_NAMES = ("manifest.json",)

# Top-level entries that may legitimately exist. Everything else that carries a
# data marker is a regression.
_ALLOWED_DATA_ROOT = "data"


def _tracked_files() -> list[str]:
    """Every file git tracks, as repo-relative POSIX paths.

    Tracked files rather than a filesystem walk, for two reasons: an untracked
    scratch ingest (`--output /tmp/...`) is a legitimate workflow and must not
    fail the suite, and a walk would trip over `node_modules/` and
    `clients/rs/target/`, which are full of unrelated `manifest.json` files.
    The damage this test exists to prevent is a *commit*.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [p for p in out.split("\0") if p]


def _carries_data(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name in _DATA_MARKER_NAMES or any(name.endswith(s) for s in _DATA_MARKERS)


def test_no_library_shaped_directory_at_repo_root() -> None:
    """No tracked parquet or manifest may live outside `data/`.

    This is the check that would have caught `hi-xs-prod/` and `tendl-2025/` the
    day they were committed. It fails on the *whole class*, not the two names:
    any future top-level `<library>/xs/*.parquet` trips it too.
    """
    offenders = sorted(
        {p.split("/", 1)[0] for p in _tracked_files() if _carries_data(p) and p.split("/", 1)[0] != _ALLOWED_DATA_ROOT}
    )
    assert not offenders, (
        f"tracked data files live outside data/, under: {offenders}.\n\n"
        "Shipped data belongs under data/ — catalog.json paths are relative to it "
        "and the release tarball is built `tar --zstd -C data`, so a top-level "
        "tree is invisible to both. This is usually an ingest run with --output "
        "pointing at the repo root (#341); re-run it with the default, or with an "
        "explicit scratch path, and `git clean` the stray tree."
    )


def test_the_guard_can_actually_fail() -> None:
    """The assertion above must reject the layout it was written for.

    A repo-scanning test that passes because it found nothing is indistinguishable
    from one whose matcher is broken. Pin the matcher against the real #341 paths
    so this cannot silently become a no-op.
    """
    regressed = ["data/tendl-2025/xs/d_Ac.parquet", "tendl-2025/xs/d_Ac.parquet", "hi-xs-prod/manifest.json"]
    offenders = sorted(
        {p.split("/", 1)[0] for p in regressed if _carries_data(p) and p.split("/", 1)[0] != _ALLOWED_DATA_ROOT}
    )
    assert offenders == ["hi-xs-prod", "tendl-2025"]


def test_release_please_manifest_is_not_mistaken_for_data() -> None:
    """`.release-please-manifest.json` is tracked at the root and must stay legal.

    It is the one root-level file whose name is close enough to `manifest.json`
    to be caught by a sloppy substring match, and flagging it would make the
    guard fire on every checkout.
    """
    assert (ROOT / ".release-please-manifest.json").exists(), "expected the release-please manifest at the repo root"
    assert not _carries_data(".release-please-manifest.json")


# -- The cause: ingest scripts must default to data/, never the repo root ------


def _parser_for(module_name: str) -> argparse.ArgumentParser:
    module = __import__(module_name)
    return module.build_parser()


@pytest.mark.parametrize("module_name", ["fetch_endf_libs", "fetch_exfor"])
def test_output_defaults_to_data_dir(module_name: str) -> None:
    """`--output` must default to `data/`, which is where every caller wants it.

    Both scripts write `<output>/<subdir>/`, so a repo-root default does not
    merely put files in an odd place — it creates a *tracked top-level directory*
    that shadows the real one under `data/`. Asserting the resolved `Path` rather
    than the help string, because the help string is not what argparse hands the
    writer.
    """
    args = _parser_for(module_name).parse_args([])
    assert args.output == DATA_DIR, f"{module_name} --output defaults to {args.output}, expected {DATA_DIR}"
    assert args.output != ROOT, f"{module_name} --output defaults to the repo root — this is the #341 regression"
    assert args.output.name == "data"


@pytest.mark.parametrize("module_name", ["fetch_endf_libs", "fetch_exfor"])
def test_output_help_names_the_default(module_name: str) -> None:
    """The help text must name the actual default.

    Both scripts said "default: repo root" while doing exactly that, so the
    surprise was accurately documented and still a surprise. The help is now an
    f-string over `DATA_DIR.name`, so this asserts the rendered result rather
    than a substring: `"default: data/"` exactly, not merely something
    containing "data/" (which `data/foo` would also satisfy).
    """
    action = next(a for a in _parser_for(module_name)._actions if "--output" in a.option_strings)
    assert action.help == f"Output directory (default: {DATA_DIR.name}/)", (
        f"{module_name} --output help does not name the default {DATA_DIR.name}/: {action.help!r}"
    )


def test_scripts_do_not_re_derive_the_data_dir() -> None:
    """`scripts/_paths.py` must remain the only place that spells `ROOT / "data"`.

    Nine scripts used to carry their own `ROOT = Path(__file__).parent.parent`
    and derive `ROOT / "data"` from it. Every one of them was *correct*, so this
    is not guarding against a live bug — it is guarding against the copy. A new
    ingest script gets written by copying an existing one, and while the
    re-derived spelling remains the majority pattern in the directory, the next
    author inherits the shape that produced #341 rather than the fixed one.

    Enforcing it here rather than in review, because "did you import DATA_DIR?"
    is exactly the kind of thing review forgets and a test never does.
    """
    offenders: dict[str, list[str]] = {}
    for script in sorted((ROOT / "scripts").glob("*.py")):
        if script.name == "_paths.py":  # the one legitimate definition
            continue
        hits = [
            f"{script.name}:{n}: {line.strip()}"
            for n, line in enumerate(script.read_text().splitlines(), 1)
            if line.lstrip().startswith(("ROOT = Path(", "ROOT=Path(")) or 'ROOT / "data"' in line
        ]
        if hits:
            offenders[script.name] = hits

    assert not offenders, (
        "scripts re-derive the repo root or the data directory instead of importing "
        "them from scripts/_paths.py:\n" + "\n".join(h for hs in offenders.values() for h in hs) + "\n\n"
        "Use `from _paths import DATA_DIR` (and `ROOT` if you genuinely need the "
        "checkout root). One place to be right is the whole point of _paths.py (#341)."
    )


def test_the_data_dir_guard_can_actually_fail() -> None:
    """The matcher above must reject the pattern it was written to retire.

    Same reasoning as `test_the_guard_can_actually_fail`: now that every script
    is clean, a broken matcher and a clean directory look identical.
    """
    legacy = ["ROOT = Path(__file__).parent.parent", '    ap.add_argument("--data-dir", default=ROOT / "data")']
    matched = [ln for ln in legacy if ln.lstrip().startswith(("ROOT = Path(", "ROOT=Path(")) or 'ROOT / "data"' in ln]
    assert matched == legacy, "the matcher no longer recognises the pre-#341 spelling"


def test_exfor_reads_tendl_from_the_data_dir() -> None:
    """`--all` enumerates elements from `data/tendl-2023-iso/`, not the repo root.

    It used to glob a repo-root `tendl-2023-iso/` that has never existed.
    `Path.glob` on a missing directory yields nothing instead of raising, so
    `--all` silently fetched zero elements — the same root-vs-`data/` confusion
    as the `--output` default, failing quietly instead of loudly.
    """
    import fetch_exfor

    assert (DATA_DIR / "tendl-2023-iso" / "xs").is_dir(), "expected data/tendl-2023-iso/xs to exist"
    assert not (ROOT / "tendl-2023-iso").exists(), "a repo-root tendl-2023-iso/ is exactly the bug"
    assert fetch_exfor.get_tendl_elements("p"), "get_tendl_elements('p') found nothing — it is globbing the wrong tree"
