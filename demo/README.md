# Demo: validating MirrorML on a small feature project

This folder is a worked example. It plays the part of a tiny real project so
you can see MirrorML do its job end to end, rather than on synthetic test
cases.

## The scenario

A team is building a customer churn model. One feature feeds it: the **average
value of each customer's valid orders**. Like most teams, they compute it
twice:

- **Offline**, in pandas, over a historical export, while training the model
  (`offline_features.py`).
- **Online**, in SQL, against the warehouse, at serving time
  (`online_features.py`).

These two pieces of code are supposed to produce the same number. MirrorML
checks that they do, and flags it when they drift apart.

## What the demo shows

`check.py` traces both pipelines and runs four checks:

1. **Correct pair** -> MirrorML reports them as equivalent (no skew).
2. **Serving sums instead of averaging** -> caught as `aggregation_function`.
3. **Serving forgot the `amount > 0` validity filter** -> caught as a
   missing-operation skew (`schema_drift`), pointed at the filter the training
   side has and serving lacks.
4. **The warehouse `ts` column is in US/Pacific, not UTC** -> caught as
   `timezone_mismatch`, even though the feature does not aggregate on `ts`.

Each of those is a mistake that ships silently in real systems: nothing
crashes, the model just quietly gets worse. MirrorML catches all three before
either pipeline runs.

## Run it

From the repository root:

```bash
uv run python demo/check.py
```

The script is self-checking: it prints a verdict for each scenario and exits
with an error code if MirrorML ever misses an injected skew or wrongly flags
the correct pair.

## What this does and does not prove

Both pipelines here were written for the demo, so this shows MirrorML working
on realistic feature logic, not on independently sourced production code. It is
a guided tour of the product, not a real-world accuracy measurement.
