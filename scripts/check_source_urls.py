"""Check that every `source_url` in `data/catalog.json` still resolves (#337).

`source_url` is the citation/provenance pointer a human follows to find where a
library came from. The ingest does not read it — `scripts/fetch_endf_libs.py`
pulls from the IAEA mirror — so nothing exercised these URLs, and one of them
rotted to NXDOMAIN without anything noticing. A dead provenance link is worse
than an absent one: it implies somebody checked.

Deliberately a script and not a test. It needs the network, and a network check
in PR CI fails on a train, on a flaky DNS resolver, and every time an upstream
institution has a bad afternoon — a check that cannot run offline is not a
check (the same argument `tests/test_library_registry.py` makes for staying
offline). Run it before a data release, when the catalog gains a URL, or when a
provenance link is about to be quoted somewhere.

Usage:
    nix develop -c uv run python scripts/check_source_urls.py
    nix develop -c uv run python scripts/check_source_urls.py --timeout 40

Exit status is 1 if any URL is dead, 0 otherwise. Redirects, soft 404s and
certificate failures are reported but do not fail the run: a redirect is usually
a site reorganising and still honouring the old path, which is fine for a
citation; the soft-404 heuristic is a hint rather than a verdict; and a chain
that will not verify from one machine is a fact about that machine and the
server's chain, not about whether the resource is there.
"""

from __future__ import annotations

import argparse
import json
import socket
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import DATA_DIR  # noqa: E402

#: Fields holding a URL a human is expected to be able to open. `base_url` is
#: excluded on purpose: it is a template containing `{version}` and is exercised
#: by the download path itself, which is a different kind of check.
URL_FIELDS = ("source_url", "mirror_url")

#: A 200 whose body says otherwise. Status code alone misses this, and a site
#: that serves a styled "page not found" with a 200 is exactly the shape that
#: outlives a reorganisation.
_SOFT_404_MARKERS = ("page not found", "404 not found", "does not exist", "page you requested")

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"


@dataclass
class Result:
    url: str
    where: list[str]
    verdict: str  # ok | redirect | soft-404 | dead
    detail: str

    @property
    def failed(self) -> bool:
        return self.verdict == "dead"


def collect_urls(catalog: dict) -> dict[str, list[str]]:
    """Every URL field in the catalog, mapped to the paths that declare it.

    Walks the whole document rather than just `libraries`, because
    `shared.stopping.compounds` carries one too and a checker that only looks
    where the last bug was is how the next one hides.
    """
    found: dict[str, list[str]] = {}

    def walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in URL_FIELDS and isinstance(value, str):
                    found.setdefault(value, []).append(f"{path}.{key}".lstrip("."))
                else:
                    walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    walk(catalog, "")
    return found


def check(url: str, where: list[str], timeout: int) -> Result:
    host = urlparse(url).hostname
    if not host:
        return Result(url, where, "dead", "not a URL")
    try:
        socket.getaddrinfo(host, None)
    except OSError as e:
        # The #337 case. Separated from an HTTP failure because it means the
        # host is gone rather than the path — nothing on that domain will work,
        # so there is no "try another path" remedy.
        return Result(url, where, "dead", f"DNS: {host} does not resolve ({e.strerror or e})")

    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(20000).decode("utf-8", "replace").lower()
            final = resp.geturl()
    except urllib.error.HTTPError as e:
        return Result(url, where, "dead", f"HTTP {e.code}")
    except Exception as e:  # noqa: BLE001
        # A certificate that will not verify is not a dead link. The resource is
        # there and the URL is right; the server ships an incomplete chain or a
        # CA this machine does not carry. Reporting it as dead would send
        # somebody off to "fix" a correct URL, so retry unverified and, if the
        # page is really served, say precisely what is wrong instead.
        if "CERTIFICATE_VERIFY_FAILED" in str(e):
            try:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                    return Result(
                        url, where, "tls", f"HTTP {resp.status} but the certificate chain does not verify from here"
                    )
            except Exception:  # noqa: BLE001
                pass
        return Result(url, where, "dead", f"{type(e).__name__}: {str(e)[:90]}")

    if any(m in body for m in _SOFT_404_MARKERS):
        return Result(url, where, "soft-404", f"200 but the body reads like an error page ({final})")
    if final.rstrip("/") != url.rstrip("/"):
        return Result(url, where, "redirect", f"-> {final}")
    return Result(url, where, "ok", "200")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser, separately from running it (#363).

    Read-only: `--catalog` names what to read and there is no output path, so
    there is nothing here that could write into the checkout by default.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", type=Path, default=DATA_DIR / "catalog.json")
    ap.add_argument("--timeout", type=int, default=30)
    return ap


def main() -> None:
    args = build_parser().parse_args()

    urls = collect_urls(json.loads(args.catalog.read_text()))
    results = [check(url, where, args.timeout) for url, where in sorted(urls.items())]

    order = {"dead": 0, "soft-404": 1, "tls": 2, "redirect": 3, "ok": 4}
    for r in sorted(results, key=lambda r: (order[r.verdict], r.url)):
        print(f"  {r.verdict:9} {r.url}")
        if r.verdict != "ok":
            print(f"            {r.detail}")
            for w in r.where:
                print(f"            declared at {w}")

    dead = [r for r in results if r.failed]
    print(
        f"\n{len(results)} URL(s): "
        + ", ".join(
            f"{n} {v}"
            for v, n in sorted(
                ((v, sum(1 for r in results if r.verdict == v)) for v in order), key=lambda kv: order[kv[0]]
            )
            if n
        )
    )
    if dead:
        print(
            "\nA dead source_url is worse than none — it says somebody checked. Find where the "
            "resource actually lives, cite how you established it, and update the catalog (#337)."
        )
    raise SystemExit(1 if dead else 0)


if __name__ == "__main__":
    main()
