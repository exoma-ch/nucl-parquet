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

from pathlib import Path

#: The repository checkout root (the parent of `scripts/`).
ROOT = Path(__file__).resolve().parent.parent

#: Where shipped data lives, and therefore where an ingest writes by default.
#:
#: Deliberately the plain checkout path rather than
#: `nucl_parquet.download.data_dir()`. That resolver answers a *reader's*
#: question ("where can I find data?") and falls back to `~/.nucl-parquet`, a
#: consumer's download cache. An ingest script writes the repo's tracked data
#: and must never silently target that cache.
DATA_DIR = ROOT / "data"
