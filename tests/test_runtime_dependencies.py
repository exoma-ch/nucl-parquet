"""Gates on what `pip install nucl-parquet` puts in a user's environment (#236).

`pycatima` sat in `[project].dependencies`, so every install pulled catima —
AGPL-3.0 — into an MIT package's runtime, while NOTICE said catima's code "is
NOT included in this distribution". Nothing in the loader imports it: only
`nucl_parquet.build_heavy_ions` does, lazily, to regenerate the
`stopping/catima_*.parquet` shards the loader then reads.

A check on license metadata would not have caught it — pycatima declares no
license at all. So the runtime set is pinned to a table of vetted licenses
instead: adding a runtime dependency means recording what it is licensed
under, here, in the same diff.

Reads `pyproject.toml` from the checkout and imports the package in a
subprocess — no data tree, no network.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())

# Every runtime dependency, with the license it was vetted under. Permissive
# only: a copyleft dependency here would bind every downstream installer.
RUNTIME_DEPENDENCY_LICENSES = {
    "duckdb": "MIT",
    "numpy": "BSD-3-Clause",
    "zstandard": "BSD-3-Clause",
}

# Needed to *build* data, never to read it. Each is copyleft or otherwise
# unfit to ship, so it lives in the `build` extra and the dev group.
BUILD_ONLY = {"pycatima": "AGPL-3.0 (catima)"}


def _names(requirements: list[str]) -> set[str]:
    return {re.split(r"[\s<>=!~\[;]", r, maxsplit=1)[0].lower() for r in requirements}


def test_every_runtime_dependency_has_a_vetted_license():
    declared = _names(PYPROJECT["project"]["dependencies"])
    assert declared == set(RUNTIME_DEPENDENCY_LICENSES), (
        "[project].dependencies no longer matches RUNTIME_DEPENDENCY_LICENSES.\n"
        f"  unvetted: {sorted(declared - set(RUNTIME_DEPENDENCY_LICENSES))}\n"
        f"  stale:    {sorted(set(RUNTIME_DEPENDENCY_LICENSES) - declared)}\n"
        "Record the license of any new runtime dependency here. If it is "
        "copyleft, it belongs in the `build` extra instead (#236)."
    )


def test_build_only_dependencies_stay_out_of_the_runtime():
    runtime = _names(PYPROJECT["project"]["dependencies"])
    build = _names(PYPROJECT["project"]["optional-dependencies"]["build"])
    dev = _names(PYPROJECT["dependency-groups"]["dev"])
    for name, license_ in BUILD_ONLY.items():
        assert name not in runtime, f"{name} ({license_}) is a runtime dependency again"
        assert name in build, f"{name} missing from the `build` extra"
        assert name in dev, f"{name} missing from the dev group, so CI cannot build with it"


def test_every_module_imports_without_the_build_only_dependencies():
    # Setting sys.modules[name] = None makes `import name` raise ImportError,
    # as if it were not installed. Importing every submodule then proves no
    # module needs it at import time — the build scripts included, since they
    # must import it lazily for the package to load at all.
    script = textwrap.dedent(
        f"""
        import importlib, pkgutil, sys
        from pathlib import Path
        for name in {sorted(BUILD_ONLY)!r}:
            sys.modules[name] = None
        sys.path.insert(0, {str(ROOT)!r})
        import nucl_parquet
        assert Path(nucl_parquet.__file__).resolve().is_relative_to({str(ROOT)!r}), nucl_parquet.__file__
        for mod in pkgutil.walk_packages(nucl_parquet.__path__, "nucl_parquet."):
            importlib.import_module(mod.name)
        """
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
