"""MirrorML CI check (template).

Copy this file to the root of your repository as ``mirrorml_check.py`` and edit
the three marked sections to point at your own offline (training) and online
(serving) feature pipelines. The MirrorML GitHub Action runs it on every pull
request and fails the build if the two pipelines diverge.

Run it locally with ``python mirrorml_check.py``. It exits 0 when the pipelines
are equivalent and 1 (printing each divergence) when they are not. This uses
MirrorML's stable Python API; nothing here depends on running the pipelines or
having production data.
"""

from __future__ import annotations

import sys

from mirrorml import diff, trace_pandas, trace_sql

# 1. Declare the input columns and their dtypes. The dtype strings come from
#    MirrorML's dtype vocabulary (docs/concepts/dtype_vocabulary.md).
SCHEMA = (("uid", "int64"), ("score", "float64"))


# 2. Your OFFLINE (training) pipeline. Here it is pandas; trace_polars works the
#    same way. source_name below must match the table the online side reads.
def offline(df):
    return df[df["score"] > 0].groupby("uid").agg({"score": "mean"})


# 3. Your ONLINE (serving) pipeline. Here it is SQL reading a table named
#    "events" (matching source_name above).
ONLINE_SQL = "SELECT uid, AVG(score) AS score FROM events WHERE score > 0 GROUP BY uid"


def main() -> int:
    offline_fp = trace_pandas(offline, input_schema=SCHEMA, source_name="events")
    online_fp = trace_sql(ONLINE_SQL, schemas={"events": SCHEMA})

    divergences = diff(offline_fp, online_fp)
    if not divergences:
        print("MirrorML: pipelines are equivalent, no training-serving skew detected.")
        return 0

    print(f"MirrorML: found {len(divergences)} divergence(s):", file=sys.stderr)
    for d in divergences:
        print(f"  - {d.category} | {d.detail}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
