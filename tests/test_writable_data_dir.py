"""A builder must never generate into the consumer download cache (#373).

`nucl_parquet.download.data_dir()` resolves $NUCL_PARQUET_DATA -> repo checkout
-> `~/.nucl-parquet/`. Sixteen modules under `nucl_parquet/` used it to decide
where to *write*. From a checkout step 2 wins, so nothing was broken in the
repo — but from an installed wheel with no environment variable, step 3 wins and
`build(data_dir=None)` silently populates the caller's download cache with
generated artefacts.

That cache is where `download()` puts *fetched release* content. #296 signs a
content manifest for it and `scripts/verify_data_release.sh` answers "is this
tree what was published?" against that manifest. Generated output landing in the
same directory makes the question unanswerable — the tree becomes a blend of
signed release data and local build output with nothing to tell them apart.

`writable_data_dir()` is `data_dir()` minus the cache step: it raises
`NoWritableDataDir` where the reader would fall back. This file gates the split
as a *class* rather than as sixteen individual fixes:

  - every module that writes resolves through the writer (static, exhaustive,
    and complete — an unclassified new module fails the suite);
  - a builder run without a checkout or an env var raises instead of writing;
  - `$HOME` is untouched when it does.

Needs no data tree and no network.
"""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nucl_parquet.download import (
    NoWritableDataDir,
    _cache_data_dir,
    _checkout_data_dir,
    _env_data_dir,
    data_dir,
    writable_data_dir,
)

#: The `download` *submodule*, fetched deliberately rather than as the attribute
#: `nucl_parquet.download`. The package re-exports a `download()` **function** of
#: the same name, so the attribute resolves to the function and shadows the
#: module. `monkeypatch.setattr("nucl_parquet.download.<x>", ...)` therefore
#: targets the wrong object, which is how the first version of these tests
#: silently patched nothing. See `test_the_download_name_is_shadowed`.
_download = importlib.import_module("nucl_parquet.download")

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "nucl_parquet"

#: The one module that resolves the data dir purely to *read* it. `loader.py`
#: opens DuckDB views over whatever tree it finds, which is exactly the case the
#: cache fallback exists to serve — a consumer with downloaded data and no
#: checkout must be able to query it.
_READERS = {"loader.py"}

#: `download.py` defines both resolvers and is the *only* legitimate writer into
#: the cache: filling it with verified release content is its entire job.
#: `__init__.py` re-exports them as package API and resolves nothing itself.
_DEFINES_THE_RESOLVERS = {"download.py", "__init__.py"}


def _modules_resolving_the_data_dir() -> dict[str, str]:
    """Map `<module>.py` -> the resolver it imports, across the package.

    Parsed rather than grepped so an aliased import (`data_dir as
    _resolve_data_dir`, which is how all sixteen were written) is matched on the
    imported *name*, not on whatever local alias it was given.
    """
    found: dict[str, str] = {}
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.endswith("download"):
                continue
            for alias in node.names:
                if alias.name in {"data_dir", "writable_data_dir"}:
                    found[path.relative_to(PKG).as_posix()] = alias.name
    return found


def _writes_parquet(path: Path) -> bool:
    return any(w in path.read_text() for w in ("write_parquet", "write_text", "write_bytes"))


# -- The split, enforced as a class -------------------------------------------


def test_every_module_that_writes_resolves_through_the_writer() -> None:
    """A module that writes must not take its destination from the reader.

    This is the whole of #373 in one assertion. It is stated over *every* module
    in the package rather than over a list of sixteen names, so a new builder
    that reaches for `data_dir()` fails here on the day it is written.
    """
    offenders = []
    for module, resolver in sorted(_modules_resolving_the_data_dir().items()):
        if module in _READERS | _DEFINES_THE_RESOLVERS:
            continue
        if _writes_parquet(PKG / module) and resolver != "writable_data_dir":
            offenders.append(f"nucl_parquet/{module} writes but resolves through {resolver}()")

    assert not offenders, (
        "modules take their write destination from the reader's resolver:\n  "
        + "\n  ".join(offenders)
        + "\n\nUse `from .download import writable_data_dir as _resolve_data_dir`. "
        "data_dir() falls back to ~/.nucl-parquet, the verified-release download "
        "cache, so writing through it corrupts the answer to 'is this tree what "
        "was published?' (#296, #373)."
    )


def test_the_reader_still_reads() -> None:
    """`loader.py` must keep the fallback — it is the case the fallback is for.

    Asserted positively so the change above cannot be "fixed" by routing
    everything through the writer, which would break querying downloaded data
    from outside a checkout — the normal way a consumer uses this package.
    """
    resolvers = _modules_resolving_the_data_dir()
    assert resolvers.get("loader.py") == "data_dir", (
        "loader.py must resolve through data_dir(): it only reads, and a consumer "
        "with a downloaded tree and no checkout has to be able to query it."
    )


def test_every_resolving_module_is_accounted_for() -> None:
    """No module may resolve the data dir without being classified.

    Same shape as the classification gate in `test_repo_layout.py`: the check
    above skips `_READERS` and `_DEFINES_THE_RESOLVERS`, so an unclassified
    non-writing module could silently sit on the reader's resolver and later grow
    a write. Force the decision when the module is added.
    """
    resolvers = _modules_resolving_the_data_dir()
    assert resolvers, "found no modules resolving the data dir — the AST scan is broken"

    unclassified = sorted(
        module
        for module, _ in resolvers.items()
        if module not in _READERS | _DEFINES_THE_RESOLVERS and not _writes_parquet(PKG / module)
    )
    assert not unclassified, (
        f"module(s) {unclassified} resolve the data dir but neither write nor are "
        "classified as readers. Add them to _READERS (they only read) or give them "
        "a write so the writer check covers them."
    )


def test_the_sixteen_are_all_still_covered() -> None:
    """The population this issue was filed about must not shrink silently.

    #373 counted the writers by hand. Pinning the number means deleting or
    renaming a builder is a deliberate act rather than something that quietly
    reduces what the check above covers.
    """
    writers = {m for m, _ in _modules_resolving_the_data_dir().items() if _writes_parquet(PKG / m)}
    assert len(writers) == 16, f"expected 16 writing modules, found {len(writers)}: {sorted(writers)}"
    # The three under g4/ were missing from the issue's list of 14; they have the
    # same shape and the same exposure.
    assert {"g4/build_all.py", "g4/ensdfstate.py", "g4/photon_evap_gammas.py"} <= writers


def test_the_download_name_is_shadowed() -> None:
    """`nucl_parquet.download` as an attribute is the function, not the module.

    `__init__.py` re-exports a `download()` function whose name collides with the
    `download.py` submodule, so attribute access returns the function. That makes
    `monkeypatch.setattr("nucl_parquet.download._checkout_data_dir", ...)` patch
    an attribute onto a *function object* — no error, no effect, and a test that
    passes while exercising nothing. The first draft of this file did exactly
    that, which is the same silent-no-op failure the rest of this suite exists to
    stamp out.

    Pinned rather than fixed: renaming either name is a breaking change to the
    package's public API, and the collision is harmless once known. This test is
    the "once known" part — if the collision is ever resolved, it fails and
    points at the workaround so it can be removed.
    """
    import nucl_parquet

    assert callable(nucl_parquet.download) and not hasattr(nucl_parquet.download, "writable_data_dir"), (
        "nucl_parquet.download is no longer shadowed by the download() function — "
        "the importlib.import_module workaround at the top of this file, and the "
        "same dance in the subprocess snippet below, can now be simplified."
    )
    assert _download.writable_data_dir is writable_data_dir, "importlib must reach the real module"


# -- The resolver itself ------------------------------------------------------


def test_writable_data_dir_agrees_with_the_reader_inside_a_checkout() -> None:
    """In the repo the two resolvers must give the same answer.

    The split is about the *fallback*, not about the normal case. If these ever
    disagree in a checkout, the refactor changed behaviour rather than narrowing
    it.
    """
    assert writable_data_dir() == data_dir() == _checkout_data_dir()


def test_scripts_paths_data_dir_is_the_same_one_answer() -> None:
    """`scripts/_paths.DATA_DIR` must *be* the writer's resolver, not a copy.

    #341 computed `ROOT / "data"` there because the only package resolver was the
    reader's. Now that a writer's exists, a second implementation that merely
    agrees is one that can stop agreeing.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import _paths

    assert _paths.DATA_DIR == writable_data_dir()


def test_env_var_is_honoured_for_writing(tmp_path, monkeypatch) -> None:
    """An explicit $NUCL_PARQUET_DATA is the caller naming their tree.

    It must work without a checkout — that is the documented escape hatch the
    error message offers, so it has to actually be one.
    """
    target = tmp_path / "mydata"
    target.mkdir()
    monkeypatch.setenv("NUCL_PARQUET_DATA", str(target))
    monkeypatch.setattr(_download, "_checkout_data_dir", lambda: None)

    assert writable_data_dir() == target
    assert _env_data_dir() == target


def test_writable_data_dir_raises_instead_of_returning_the_cache(tmp_path, monkeypatch) -> None:
    """The #373 scenario: installed wheel, no env var, cache present.

    The reader returns the cache — correct for it. The writer must refuse, and
    the message must say why rather than just "not found", because the caller's
    next move (pass data_dir=, set the env var, use a checkout) is not obvious
    from the failure alone.
    """
    cache = tmp_path / "home" / ".nucl-parquet"
    cache.mkdir(parents=True)
    monkeypatch.delenv("NUCL_PARQUET_DATA", raising=False)
    monkeypatch.setattr(_download, "_checkout_data_dir", lambda: None)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))

    # The reader is unchanged and still resolves to the cache.
    assert data_dir() == cache
    assert _cache_data_dir() == cache

    with pytest.raises(NoWritableDataDir) as excinfo:
        writable_data_dir()

    message = str(excinfo.value)
    assert str(cache) in message, "the error should name the directory it refused"
    assert "NUCL_PARQUET_DATA" in message and "data_dir=" in message, "it should say how to fix it"


def test_writable_data_dir_raises_when_there_is_no_cache_either(tmp_path, monkeypatch) -> None:
    """With nothing to fall back to, it still raises the writer's error.

    Not `FileNotFoundError`: "you have no data" and "you may not build here" are
    different conditions and a caller may reasonably handle them differently.
    """
    monkeypatch.delenv("NUCL_PARQUET_DATA", raising=False)
    monkeypatch.setattr(_download, "_checkout_data_dir", lambda: None)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "empty-home"))

    with pytest.raises(NoWritableDataDir):
        writable_data_dir()


# -- End to end: a real builder, a real subprocess, a real $HOME ---------------

_BUILDER_REFUSES = """
import sys, pathlib
sys.path.insert(0, {root!r})

import importlib
# The package re-exports a `download()` function, so `nucl_parquet.download`
# as an attribute is that function, not this module. Fetch the module itself.
dl = importlib.import_module("nucl_parquet.download")
# Simulate an installed wheel: the package is importable but there is no
# data/catalog.json beside it, which is exactly what _checkout_data_dir probes.
dl._checkout_data_dir = lambda: None

from nucl_parquet.{module} import {entry} as build

try:
    build(data_dir=None)
except dl.NoWritableDataDir as exc:
    print("REFUSED")
    sys.exit(0)
except Exception as exc:
    print("WRONG-ERROR:" + type(exc).__name__ + ":" + str(exc)[:200])
    sys.exit(2)
print("WROTE-SOMETHING")
sys.exit(3)
"""

# One representative per resolution shape. build_epdl is the heaviest writer (10
# write sites) and g4/build_all is the one the issue's list missed.
_END_TO_END_BUILDERS = [
    ("build_epdl", "build"),
    ("build_xcom", "build"),
    ("build_em_stopping", "build"),
    ("build_beta_spectra", "build"),
    ("g4.build_all", "build_all"),
]


@pytest.mark.parametrize(("module", "entry"), _END_TO_END_BUILDERS)
def test_builder_refuses_and_leaves_home_untouched(module: str, entry: str, tmp_path: Path) -> None:
    """The check the issue asked for, run for real in a subprocess.

    $NUCL_PARQUET_DATA unset, no checkout on the path, `$HOME` pointed at an
    empty directory. The builder must raise — and, more importantly, `$HOME` must
    still be empty afterwards, because "it raised" and "it wrote nothing" are
    different claims and only the second one is the actual requirement.
    """
    home = tmp_path / "home"
    home.mkdir()
    cache = home / ".nucl-parquet"
    cache.mkdir()  # the fallback exists and is therefore temptingly resolvable

    env = {
        "HOME": str(home),
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for passthrough in ("LD_LIBRARY_PATH", "NIX_LD", "NIX_LD_LIBRARY_PATH", "PYTHONPATH"):
        if passthrough in os.environ:
            env[passthrough] = os.environ[passthrough]

    result = subprocess.run(
        [sys.executable, "-c", _BUILDER_REFUSES.format(root=str(ROOT), module=module, entry=entry)],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )

    assert "REFUSED" in result.stdout, (
        f"nucl_parquet.{module}.{entry}(data_dir=None) did not raise NoWritableDataDir "
        f"with no checkout and no $NUCL_PARQUET_DATA.\nstdout: {result.stdout}\nstderr: {result.stderr[-2000:]}"
    )
    assert list(cache.iterdir()) == [], f"the builder wrote into the download cache: {list(cache.iterdir())}"
