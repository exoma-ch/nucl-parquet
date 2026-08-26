"""Repo-layout paths shared by the ingest scripts (#341).

Every `fetch_*.py` / `build_*.py` needs the same two answers — where is the
checkout, and where does data live — and each one used to spell them itself.
Two of them spelled the second answer wrong:

    scripts/fetch_endf_libs.py:  --output default=ROOT   -> <root>/<library>/xs/
    scripts/fetch_exfor.py:      --output default=ROOT   -> <root>/exfor/

Both write `<output>/<subdir>/`, so with the repo root as the default a plain
re-ingest scattered fresh parquets into *top-level* directories rather than into
`data/`. That is how `hi-xs-prod/` and `tendl-2025/` came to exist at the root
shadowing their `data/` counterparts, and it bit the #334 re-ingest, which had to
be undone with `git checkout` + `git clean`.

`DATA_DIR` exists so there is one place to be right. Import it rather than
re-deriving `ROOT / "data"`:

    sys.path.insert(0, str(Path(__file__).parent))
    from _paths import DATA_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

#: The repository checkout root (the parent of `scripts/`).
ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))  # so a bare `python scripts/foo.py` finds the package

from nucl_parquet.download import writable_data_dir  # noqa: E402

#: Where an ingest writes by default.
#:
#: This *delegates* to `nucl_parquet.download.writable_data_dir()` rather than
#: computing `ROOT / "data"` itself, which reverses the note that stood here
#: since #341. The reason given then was that the only package-level resolver was
#: `data_dir()`, a *reader's* answer that falls back to `~/.nucl-parquet` — a
#: consumer's download cache an ingest must never target. That objection was
#: about the fallback, and #373 removed it: `writable_data_dir()` is the same
#: resolution minus the cache step, raising instead.
#:
#: With the objection answered, computing the path here as well would leave two
#: implementations of "where does a writer put data" that merely happen to agree.
#: One of them would eventually stop agreeing. So there is now exactly one, and
#: `tests/test_writable_data_dir.py` pins that this constant *is* it.
#:
#: Consequence worth knowing: `$NUCL_PARQUET_DATA` now moves the scripts' default
#: too, where before they always targeted the checkout. That is the environment
#: variable's documented meaning — "this is my data tree" — and every script
#: takes an explicit `--output` / `--data-dir` (#363) when you want otherwise.
DATA_DIR = writable_data_dir()
