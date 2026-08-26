# MF=10 oracle fixtures

Real ENDF-6 evaluations, cut down to their MF=10 sections, used by
`tests/test_mt_residuals.py` to check that `mt_to_residual` names the same
product the evaluator does.

## Why these exist

`MT_EMITTED_PARTICLES` in `scripts/fetch_endf_libs.py` says which particles a
reaction channel emits; `mt_to_residual` subtracts them from target + projectile
to name the product. Nothing in the repo could check that answer, so 13 of ~30
entries carried the wrong (Z, A) for years and every affected row was filed
under the wrong nuclide — not dropped, *misattributed*, so the row looked fine
and the cross-section was a plausible number (#351).

MF=10 breaks the circle. It names its product **directly**, as
`IZAP = Z*1000 + A`, so for an MF=10 section at MT=X whose levels all name one
product, that IZAP *is* the residual of reaction MT=X, written down by the
evaluator. It is an oracle precisely because it is not derived from MT.

That only holds if the numbers are the evaluator's. Every retained line here is
byte-for-byte from the IAEA mirror; nothing was reformatted, resampled or
hand-entered. A synthetic fixture would have been written from the same table
the test checks, and would have agreed with any table, right or wrong.

## What is in each file

Only MF=10 sections, plus the original TPID line and the structural
SEND/FEND/MEND/TEND records needed to make a parseable material. MF=1's
descriptive text alone is three times the size of everything kept, and the
oracle does not read it. The MF=10 head record carries `ZA`, so each fixture
still names its own target.

| file | target | MF=10 MTs | why this one |
|---|---|---|---|
| `jeff-4.0/n_049-In-114_4928.endf` | In-114 | 4, 11, 16, 17, 22, 24, 25, 28, 29, 33, 34, 37, 42, 45, 104, 107, 111, 116, 117 | widest single-file coverage; witnesses MT=11, absent from the table before #351 |
| `fendl-3.2/n_024-Cr-50_2425.endf` | Cr-50 | 5, 17, 22, 23, 24, 25, 29, 30, 35, 36, 37, 42, 45, 108, 109, 112, 114 | the only source in the sample for MT=23, 30, 35, 36, 109 and 114 — the exotic channels, and five of the wrong entries |
| `jeff-4.0/n_033-As-75_3325.endf` | As-75 | 32, 41, 44, 103, 105, 106, 108, 111, 112, 115 | MT=44, 111 and 115, three more of the wrong entries |
| `tendl-2025/n_032-Ge-76_3243.endf` | Ge-76 | 11, 16, 24, 32, 33, 37, 41, 42, 105, 108 | a third library, and an independent witness for MT=11 and 37 |

Together they cover 34 of the 35 MT numbers that any MF=10 section named a
single product for in a 74-evaluation sample across 8 libraries. The exception
is **MT=102** (capture): the only MF=10 witness for it in the sample is
IRDFF-II's In-115, whose MF=10 sections are 1.7 MiB — 17× this whole directory
— for one entry whose emission is `("g",)`, i.e. (0, 0). It is covered by the
`endf.reaction.REACTION_NAME` check in the same test file instead.

## Regenerating

```console
$ nix develop -c uv run python tests/fixtures/mf10/extract_fixtures.py
```

Needs network. The script is deterministic: re-running it against an unchanged
mirror reproduces these files byte-for-byte. Adding a source means adding a row
to `SOURCES` there and a row to the table above.
