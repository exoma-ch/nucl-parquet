# Cross-language golden-file parity tests

Per **#176 / ADR-0002 additive policy**. Pins the cross-language API surface against a small set of canonical fixtures so the three clients (Python / Rust / TS) don't drift on NULL semantics, dtype mapping, or ordering.

## Layout

```
tests/golden/
├── README.md                       (this file)
├── fixtures/                       (JSON files — committed)
│   ├── co60_beta_gamma.json
│   ├── identify_gamma_1173keV.json
│   └── ...                         (4 more per #176 scope)
├── generate.py                     (regenerate fixtures from Python loader = source of truth)
├── normalize.py                    (Python normalization helper)
├── run_python.py                   (Python test driver)
└── (Rust + TS goldens live in their language's tests/ directory)
```

## Adding a new fixture

1. Add the canonical query to `generate.py`.
2. Run `uv run python tests/golden/generate.py` to regenerate.
3. Eyeball the diff. If it looks right, commit the fixture JSON.
4. CI runs Python / Rust / TS golden tests against the same fixture.

## Normalization rules

Per `docs/NULL_CONVENTIONS.md`:

- Sort rows by the table's documented stable key before diff.
- Round floats to 6 decimal places (specific columns enumerated per-table in `normalize.*`).
- NULL → JSON `null`. Empty-string historic columns (`state`, `parent_state`) stay as `""`.
- `+inf` half-life round-trips as the JSON string `"Infinity"` (JSON can't represent infinity natively; helpers translate).

## CI gate

`.github/workflows/ci.yml` runs the three golden test suites on every PR; non-empty diff = test failure.
