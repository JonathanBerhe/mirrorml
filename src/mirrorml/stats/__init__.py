"""Statistical companion check — planned for a future milestone.

The static fingerprint check is the primary equivalence test. For cases
where two pipelines have different fingerprints but might still be
semantically equivalent (UDF bodies that compute the same value, two
different formulations of the same aggregation), a statistical companion
runs both pipelines on a shared fixture and compares outputs within
tolerance.

This module is intentionally empty in v0.0.1.
"""

from __future__ import annotations
