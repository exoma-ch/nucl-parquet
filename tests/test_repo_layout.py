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

Needs only a git checkout — no data tree, no network.
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
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


def _prose_lines(source: str) -> set[int]:
    """Line numbers occupied by comments or docstrings.

    The matcher below is textual, so a docstring that *describes* the retired
    spelling reads exactly like the spelling itself — this very file's
    explanation of the bug tripped its own guard. Excluding prose keeps the
    check on code without weakening it.
    """
    prose: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            prose.update(range(tok.start[0], tok.end[0] + 1))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is None:
                continue
            expr = node.body[0]
            prose.update(range(expr.lineno, (expr.end_lineno or expr.lineno) + 1))
    return prose


def _re_derives_a_repo_path(line: str) -> bool:
    """Does this source line reconstruct the checkout root or the data dir itself?

    Three spellings, because the first version of this matcher only knew the
    first two and two scripts were quietly using the third.
    """
    stripped = line.lstrip()
    return (
        stripped.startswith(("ROOT = Path(", "ROOT=Path("))
        or 'ROOT / "data"' in line
        # `Path(__file__).parent.parent / "data"` inline, and `repo_root / "data"`
        # — same derivation, no variable named ROOT anywhere in sight.
        or re.search(r'Path\(__file__\)(\.parent){2,}\s*/\s*"data"', line) is not None
        or re.search(r'\brepo_root\s*/\s*"data"', line) is not None
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
        source = script.read_text()
        prose = _prose_lines(source)
        hits = [
            f"{script.name}:{n}: {line.strip()}"
            for n, line in enumerate(source.splitlines(), 1)
            if n not in prose and _re_derives_a_repo_path(line)
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
    """The matcher above must reject every spelling it was written to retire.

    Same reasoning as `test_the_guard_can_actually_fail`: now that every script
    is clean, a broken matcher and a clean directory look identical.

    The last two entries are the ones the original matcher **missed**. It keyed on
    the name `ROOT`, so `Path(__file__).parent.parent / "data"` — written out
    inline, never binding `ROOT` at all — sailed past it. Two scripts were using
    exactly that (`update_suppliers.py`, `build_kerma.py`) and the #349 guard
    reported the directory clean. A matcher that only catches the spelling you
    happened to write is the same silent-pass failure as the rest of this file.
    """
    legacy = [
        "ROOT = Path(__file__).parent.parent",
        '    ap.add_argument("--data-dir", default=ROOT / "data")',
        'SUPPLIERS_PATH = Path(__file__).parent.parent / "data" / "suppliers.json"',
        '        data_dir = repo_root / "data"',
    ]
    matched = [ln for ln in legacy if _re_derives_a_repo_path(ln)]
    assert matched == legacy, "the matcher no longer recognises the pre-#341 spelling"


# -- Every script must be introspectable and must not write by surprise (#363) --
#
# `fetch_iupac_compositions.py` had no argparse at all: `main()` was a
# parameterless side effect that fetched from NIST and overwrote a *tracked*
# parquet. It fired during #349, when a probe called `main()` on each ingest
# module to compare path constants — every sibling exposed `build_parser()` and
# was inspected without side effects; this one ran. It happened to be harmless
# because the builder is deterministic, which is a property of the data, not of
# the script.
#
# Every scripts/*.py is classified below. A new script that is in none of these
# buckets fails `test_every_script_is_classified`, so the class check cannot go
# stale by someone adding a script and not thinking about it.

#: script -> the argparse dest that decides where it writes. Must exist, be
#: Path-typed, and (when it has a default) default inside the data directory.
_OUTPUT_DEST = {
    "backfill_xs_nuclides.py": "data_dir",
    "build_channels.py": "out_dir",
    "build_kerma.py": "data_dir",
    "build_manifests.py": "data_dir",
    "build_neutron_njoy.py": "out",
    "build_neutron_total.py": "data_dir",
    "fetch_ame2020.py": "output",
    "fetch_endf_libs.py": "output",
    "fetch_exfor.py": "output",
    "fetch_iupac_compositions.py": "output",
    "fetch_strata_nuclear.py": "dest_dir",
    "migrate_state_vocabulary.py": "data_dir",
    "migrate_xs_schema.py": "data_dir",
    "update_suppliers.py": "data_dir",
}

#: Writes only where the caller points it, with no default at all — the strongest
#: form of "you chose this". `fetch_exfor_master` takes `out` as a required
#: positional; `fetch_stsv` writes nothing unless `--out` is given.
_OUTPUT_REQUIRED = {"fetch_exfor_master.py": "out", "fetch_stsv.py": "out"}

#: Writes to a place that is not a data path: `sync_huggingface` pushes to the
#: remote named by `--repo-id` (and has `--dry-run`); `build_readme` rewrites
#: README.md and only when `--write` is passed.
_WRITES_ELSEWHERE = {"sync_huggingface.py", "build_readme.py"}

#: Audits and reports; writes nothing.
_READ_ONLY = {
    "check_builder_staleness.py",
    "check_isomeric_sum_rule.py",
    "check_source_urls.py",
}

#: Shared helper modules, not runnable scripts — underscore-prefixed by
#: convention. `_canonical.py` (#359/#366) holds the canonical row vocabulary and
#: frame transform; it was caught by `test_every_script_is_classified` when this
#: branch rebased onto it, which is the guard working as intended on a real file
#: rather than a synthetic one.
_NOT_A_SCRIPT = {"_paths.py", "_canonical.py"}


def _script_names() -> set[str]:
    return {p.name for p in (ROOT / "scripts").glob("*.py")}


def _module_for(script_name: str):
    return __import__(script_name[: -len(".py")])


def test_every_script_is_classified() -> None:
    """A new script must be placed in one of the buckets above.

    Without this, the checks below silently stop covering the directory the
    moment someone adds a file — the same way `scripts/ci.sh`'s allowlist stops
    covering a new test file.
    """
    classified = set(_OUTPUT_DEST) | set(_OUTPUT_REQUIRED) | _WRITES_ELSEWHERE | _READ_ONLY | _NOT_A_SCRIPT
    unclassified = sorted(_script_names() - classified)
    stale = sorted(classified - _script_names())
    assert not unclassified, (
        f"new script(s) {unclassified} are not classified in tests/test_repo_layout.py. "
        "Add each to _OUTPUT_DEST (it writes into the data dir), _OUTPUT_REQUIRED "
        "(the caller must name the destination), _WRITES_ELSEWHERE, or _READ_ONLY."
    )
    assert not stale, f"classified script(s) {stale} no longer exist — drop them from the table"


@pytest.mark.parametrize("script_name", sorted(_script_names() - _NOT_A_SCRIPT))
def test_every_script_exposes_build_parser(script_name: str) -> None:
    """Importing a script must reveal its CLI without running it.

    `build_parser()` is the seam that makes a script inspectable — what are its
    defaults, where does it write — without the import itself doing anything. A
    script whose only entry point is `main()` can only be interrogated by
    executing it, which for an ingest means a network fetch and a file write.
    """
    module = _module_for(script_name)
    assert hasattr(module, "build_parser"), (
        f"scripts/{script_name} does not expose build_parser(). Extract the "
        "ArgumentParser out of main() so its defaults can be read without executing it (#363)."
    )
    parser = module.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)


@pytest.mark.parametrize("script_name", sorted(_OUTPUT_DEST))
def test_output_location_is_overridable_and_defaults_into_the_data_dir(script_name: str) -> None:
    """Where a script writes must be a CLI argument, defaulting under `data/`.

    Two failure modes in one assertion. Not overridable at all is #363
    (`fetch_iupac_compositions` wrote a module constant). Overridable but
    defaulting to the repo root is #341 (`fetch_endf_libs --output` scattered a
    623-file tree next to `data/`).
    """
    dest = _OUTPUT_DEST[script_name]
    parser = _module_for(script_name).build_parser()
    action = next((a for a in parser._actions if a.dest == dest), None)
    assert action is not None, f"scripts/{script_name} has no --{dest.replace('_', '-')} argument"
    assert action.type is Path, f"scripts/{script_name} --{dest.replace('_', '-')} is not Path-typed"

    default = action.default
    assert isinstance(default, Path), f"scripts/{script_name} --{dest.replace('_', '-')} has no Path default"
    assert default == DATA_DIR or DATA_DIR in default.parents, (
        f"scripts/{script_name} --{dest.replace('_', '-')} defaults to {default}, which is outside "
        f"{DATA_DIR}. An ingest default outside the data directory is the #341 regression."
    )
    assert default != ROOT, f"scripts/{script_name} defaults to the repo root — the #341 regression exactly"


@pytest.mark.parametrize("script_name", sorted(_OUTPUT_REQUIRED))
def test_scripts_without_an_output_default_require_one(script_name: str) -> None:
    """No default is fine — silently defaulting somewhere surprising is not.

    These two write only where told. Pinned so nobody "helpfully" gives them a
    default later without going through the check above.
    """
    dest = _OUTPUT_REQUIRED[script_name]
    parser = _module_for(script_name).build_parser()
    action = next((a for a in parser._actions if a.dest == dest), None)
    assert action is not None, f"scripts/{script_name} lost its {dest} argument"
    assert action.default is None, f"scripts/{script_name} {dest} acquired a default: {action.default}"


@pytest.mark.parametrize(
    "script_name",
    [
        "fetch_iupac_compositions.py",
        "fetch_ame2020.py",
        "update_suppliers.py",
        "migrate_xs_schema.py",
        # Fetches ENDF tapes from the IAEA mirror and rewrites tracked shards
        # in place — the sharpest form of this class, since it *merges* into an
        # existing file rather than replacing it (#335).
        "backfill_xs_nuclides.py",
    ],
)
def test_network_writers_offer_a_dry_run(script_name: str) -> None:
    """A script that fetches and then overwrites tracked data must offer a no-write mode.

    The three fetchers here had no `--dry-run`, so "see what this would do" and
    "do it" were the same command. That is what turned a read-only inspection
    during #349 into a write.
    """
    parser = _module_for(script_name).build_parser()
    assert any(a.dest == "dry_run" for a in parser._actions), (
        f"scripts/{script_name} fetches and overwrites tracked data but has no --dry-run (#363)"
    )
    assert parser.parse_args([]).dry_run is False, "--dry-run must be opt-in"


def test_no_ingest_script_uses_the_reader_side_data_dir_resolver() -> None:
    """`nucl_parquet.download.data_dir()` must not decide where a script writes.

    It resolves $NUCL_PARQUET_DATA -> checkout -> **`~/.nucl-parquet/`**. That
    last step is a consumer's download cache, so an ingest defaulting to it would
    quietly populate the cache of whoever ran it instead of failing. Right for a
    reader ("find the data, wherever it is"), wrong for a writer (which means one
    specific tree).

    #349 deliberately built `scripts/_paths.DATA_DIR` on the plain checkout path
    for this reason and wrote down why, but a docstring does not stop the next
    import. This does. Deliberately *not* solved by adding a writer-side resolver
    to `nucl_parquet`: that would be a third spelling of "where does data live"
    in a project whose first representation principle is one spelling per
    concept, and the boundary it needs to defend is scripts/, which is here.
    """
    offenders = []
    for script in sorted((ROOT / "scripts").glob("*.py")):
        source = script.read_text()
        prose = _prose_lines(source)
        for n, line in enumerate(source.splitlines(), 1):
            if n in prose:
                continue
            if re.search(r"\b(from nucl_parquet\.download import|download\.data_dir|_resolve_data_dir)\b", line):
                offenders.append(f"{script.name}:{n}: {line.strip()}")

    assert not offenders, (
        "ingest scripts must not resolve their output path with the reader-side "
        "nucl_parquet.download.data_dir():\n" + "\n".join(offenders) + "\n\n"
        "Use `from _paths import DATA_DIR`. data_dir() falls back to ~/.nucl-parquet, "
        "a download cache, so writing through it targets the wrong tree outside a checkout."
    )


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
