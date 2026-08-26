"""CI must run the test suite, not a hand-picked subset of it (#355).

`scripts/ci.sh` used to invoke pytest with an explicit list of test files. A
list is an allowlist, and a test file not named on it never executed — silently,
with a green tick. **40 of 54 test files were in that state.** Among them
`test_readme_drift.py`, the check `CLAUDE.md` promises will fail the suite if
you forget to regenerate the README; in CI it did not exist.

This is the failure shape of #334, #340, #341 and #351 relocated into the
harness itself: a lookup misses, the default is benign, the run exits 0. And it
is not hypothetical — it bit two pull requests on a single day. #341 added
`test_repo_layout.py`, the gate written specifically to stop a 623-file stale
tree from coming back, and it did not run in CI until a follow-up commit. #354
added `test_fetch_endf_libs.py` and had to append it to `ci.sh` by hand. Both
authors caught it. Neither was caught by the repository.

So the guard is on the harness. Two invariants, because a list can come back in
two different ways:

1. `ci.sh` names no individual test file — it runs the directory.
2. Every `tests/**/test_*.py` actually yields collected tests, so a file cannot
   quietly become dead weight through a rename, an import error, or a module
   that defines nothing pytest recognises.

Runs offline against the checkout; needs no data tree and no network.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).parent
_ROOT = _TESTS_DIR.parent
_CI_SH = _ROOT / "scripts" / "ci.sh"

#: A bare `pytest` word, not a prefix pattern.
#:
#: Matching `^\s*(?:uv run )?pytest` would have been the obvious thing and is
#: wrong in the direction that matters: `python -m pytest …`, `uvx pytest …`,
#: `PYTHONPATH=. pytest …` and `env FOO=bar pytest …` all invoke pytest and none
#: of them start the line with it. A second invocation written any of those ways
#: would evade every check below while the canonical one kept them green — the
#: guard passing by not looking, which is the whole complaint this file is
#: about. So: find the token, wherever on the line it sits, and let the caller
#: decide what the line is doing.
_PYTEST_TOKEN = re.compile(r"(?<![\w.-])pytest(?![\w.-])")

#: Test files anywhere under `tests/`, including `tests/g4/` and
#: `tests/golden/`. `tests/test_\w+\.py` would have missed the subdirectories.
_TEST_FILE_ARG = re.compile(r"tests(?:/\w+)*/test_\w+\.py")


def _pytest_command_lines() -> list[str]:
    """Every logical shell line in `ci.sh` that invokes pytest.

    Comments are dropped first (`ci.sh` discusses test files in prose, at
    length, and that prose is not an invocation), then backslash continuations
    are joined so a multi-line invocation is examined whole.
    """
    lines, buf = [], ""
    for raw in _CI_SH.read_text().splitlines():
        if raw.lstrip().startswith("#"):
            continue
        buf += raw
        if raw.rstrip().endswith("\\"):
            buf = buf.rstrip()[:-1] + " "
            continue
        lines.append(buf)
        buf = ""
    if buf:
        lines.append(buf)
    return [ln for ln in lines if _PYTEST_TOKEN.search(ln)]


def test_ci_invokes_pytest_at_all() -> None:
    """Positive assertion first.

    Every check below is of the form "ci.sh does not do X". All of them pass
    vacuously against a `ci.sh` that has no pytest in it at all, which is a
    worse state than the one this file exists to prevent.
    """
    invocations = _pytest_command_lines()
    assert invocations, f"{_CI_SH} contains no pytest invocation — the suite is not running in CI at all"


def test_ci_runs_the_tests_directory_not_a_list_of_files() -> None:
    """The invariant #355 is about.

    Naming files is how the allowlist got here: each new gate was appended by
    hand, and the ones nobody remembered to append simply never ran. Running the
    directory makes that unrepresentable — a new test file is picked up because
    it exists, not because someone remembered it.
    """
    named = sorted({f for inv in _pytest_command_lines() for f in _TEST_FILE_ARG.findall(inv)})
    assert not named, (
        "scripts/ci.sh names individual test files in a pytest invocation:\n  "
        + "\n  ".join(named)
        + "\n\nRun `pytest tests/` and let the markers filter. A named list is an allowlist, and a "
        "gate nobody remembers to add to it never runs (#355)."
    )


def test_ci_runs_pytest_exactly_once_over_the_suite() -> None:
    """One invocation, so there is one answer to 'what does CI run?'.

    #358 needed a second invocation to claw `test_manifests.py` back from the
    `-m "not data"` filter. That was a local patch on a general problem, and two
    invocations means two filters to keep in step — the next gate lands under
    whichever one its author happened to read.
    """
    invocations = _pytest_command_lines()
    assert len(invocations) == 1, (
        f"scripts/ci.sh has {len(invocations)} pytest invocations; expected 1 over tests/:\n  "
        + "\n  ".join(inv.strip() for inv in invocations)
    )


def test_ci_does_not_filter_out_the_data_marker() -> None:
    """`data` is a degradation mechanism, not a CI opt-out.

    `tests/conftest.py` skips `data`-marked tests when the data tree is absent,
    which is the entire job that marker has. The tree is committed, so filtering
    them out in CI only hid checks that had something to say: `exfor-channels`'
    manifest disagreed with its own parquets from #334 until #358, and the check
    that would have said so was deselected.
    """
    for inv in _pytest_command_lines():
        assert "not data" not in inv, (
            f"scripts/ci.sh deselects the `data` marker:\n  {inv.strip()}\n\n"
            "conftest.py already skips those tests when the data tree is absent; the tree is "
            "committed, so this only suppresses checks on the shipped data (#355)."
        )


def test_every_test_file_yields_collected_tests() -> None:
    """No `tests/test_*.py` may be dead weight.

    Collected in a subprocess with no marker filter, so this answers "does the
    file contribute tests at all", independent of whatever filter the current
    session is running under. A file that fails to import shows up as a
    collection error and fails here rather than being absent-and-unnoticed —
    which is the whole complaint.
    """
    # rglob, not glob: tests/g4/ and tests/golden/ hold test files too, and a
    # top-level-only check would have declared them covered without looking.
    on_disk = {p.relative_to(_TESTS_DIR).as_posix() for p in sorted(_TESTS_DIR.rglob("test_*.py"))}
    assert len(on_disk) >= 40, f"only {len(on_disk)} test files found — is {_TESTS_DIR} right?"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(_TESTS_DIR),
            "--collect-only",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    # `--collect-only -q` prints node ids rooted at the invocation cwd (_ROOT),
    # e.g. `tests/g4/test_xray_auger.py::test_x`. Normalise to the same
    # tests/-relative form as `on_disk` so a subdirectory cannot look missing.
    collected = set()
    for line in proc.stdout.splitlines():
        if "::" not in line:
            continue
        path = line.split("::", 1)[0].strip()
        # Fail loudly rather than guessing. Silently stripping to a basename
        # here would let a subdirectory file match a top-level one of the same
        # name, and the guard would report success on the wrong comparison.
        assert path.startswith("tests/"), (
            f"unexpected node id shape from --collect-only: {line!r}. Expected paths rooted at "
            f"{_ROOT} (cwd of the subprocess); the normalisation below assumes it."
        )
        collected.add(path[len("tests/") :])

    silent = sorted(on_disk - collected)
    assert not silent, (
        "these test files are on disk but yield no collected tests:\n  "
        + "\n  ".join(silent)
        + "\n\nA file that collects nothing is dead weight that reads like a guard (#355). "
        f"\n\ncollect-only exit={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )
    assert proc.returncode == 0, (
        f"`pytest --collect-only tests/` exited {proc.returncode} — a collection error means a test "
        f"file cannot even be imported:\n{proc.stdout[-3000:]}\n{proc.stderr[-2000:]}"
    )


def test_this_guard_would_notice_an_uncollectable_file(tmp_path: Path) -> None:
    """Prove the collection check has teeth rather than assuming it.

    A guard that reports success by finding nothing is the defect this whole
    file is about, so the mechanism gets exercised on a directory where the
    answer is known: one importable test file and one that raises on import.
    """
    (tmp_path / "test_fine.py").write_text("def test_ok():\n    assert True\n")
    (tmp_path / "test_broken.py").write_text("import a_module_that_does_not_exist  # noqa: F401\n")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tmp_path),
            "--collect-only",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    collected = {line.split("::", 1)[0].rsplit("/", 1)[-1] for line in proc.stdout.splitlines() if "::" in line}

    assert "test_fine.py" in collected, f"the importable file was not collected:\n{proc.stdout}"
    assert "test_broken.py" not in collected
    assert proc.returncode != 0, "an unimportable test file must make --collect-only fail"


@pytest.mark.parametrize("marker", ["data", "network"])
def test_declared_markers_are_the_only_ones_used(marker: str) -> None:
    """Markers are the filtering mechanism now, so the vocabulary must be closed.

    `pyproject.toml` declares `data` and `network`. A misspelt marker attribute
    silently marks nothing, and without `--strict-markers` it does not even
    warn — the test would then run in CI and try to reach the network.
    """
    declared = (_ROOT / "pyproject.toml").read_text()
    assert f'"{marker}:' in declared, f"marker {marker!r} is used by the suite but not declared in pyproject.toml"


def test_no_test_file_uses_an_undeclared_marker() -> None:
    """The other direction: nothing in tests/ may mark itself with a marker
    pytest does not know about, because pytest would ignore it."""
    declared = set(re.findall(r'^\s*"(\w+):', (_ROOT / "pyproject.toml").read_text(), re.MULTILINE))
    builtin = {"parametrize", "skip", "skipif", "xfail", "usefixtures", "filterwarnings"}
    unknown: dict[str, set[str]] = {}
    for path in sorted(_TESTS_DIR.rglob("test_*.py")):
        used = set(re.findall(r"pytest\.mark\.(\w+)", path.read_text()))
        stray = used - declared - builtin
        if stray:
            unknown[path.relative_to(_TESTS_DIR).as_posix()] = stray
    assert not unknown, (
        "test files use markers that pyproject.toml does not declare (pytest ignores these silently):\n  "
        + "\n  ".join(f"{f}: {sorted(m)}" for f, m in unknown.items())
    )
