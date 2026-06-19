# MirrorML GitHub Action

Run MirrorML in CI to catch training-serving skew before it merges. On every
pull request that touches feature code, the action traces your offline
(training) and online (serving) pipelines and fails the build if they diverge.

## Setup (two files in your repository)

1. **`mirrorml_check.py`** at your repo root. Copy
   [`mirrorml_check.py`](./mirrorml_check.py) and edit the three marked sections
   to point at your own pipelines. It uses MirrorML's stable Python API
   (`trace_pandas` / `trace_polars` / `trace_sql` + `diff`); it exits 0 when the
   pipelines are equivalent and 1 (listing each divergence) when they are not.

2. **`.github/workflows/mirrorml.yml`**. Copy [`ci.yml`](./ci.yml).

That is the whole setup. The default action command is `python mirrorml_check.py`.

## Inputs

| Input | Default | Notes |
|---|---|---|
| `command` | `python mirrorml_check.py` | The check to run. Can also be the CLI, e.g. `mirrorml verify path/to/pairs`. |
| `python-version` | `3.12` | Python to set up. |
| `extras` | `pandas,polars` | MirrorML extras to install. Use empty for SQL-only checks. |
| `version` | latest | PyPI version of `mirrorml` to install. Pin for reproducible CI. |

## One thing to get right

In your check script, `source_name` (on `trace_pandas` / `trace_polars`) must
match the table the SQL side reads from. MirrorML treats the source's name as
part of the pipeline's identity, so a mismatch is reported as a difference. The
template uses `events` on both sides; change both together.

## Local run

The check is just a script, so you can run it the same way CI does before
pushing:

```bash
python mirrorml_check.py
```
