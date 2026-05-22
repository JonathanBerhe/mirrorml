"""Schema version migration. Stub in v1.0.0 — only one version exists.

When the schema version advances, register an upgrade path here that takes a
raw ``dict`` of one version and returns one shaped for the latest
:class:`~mirrorml.fingerprint.schema.Fingerprint`. Cross-version loading must
always go through this module; never silently load a fingerprint of a
different version.
"""

from __future__ import annotations

from typing import Any

from mirrorml.exceptions import FingerprintVersionError
from mirrorml.fingerprint.schema import (
    MIN_SUPPORTED_SCHEMA_VERSION,
    SCHEMA_VERSION,
    Fingerprint,
)


def migrate(raw: dict[str, Any], target: str = SCHEMA_VERSION) -> Fingerprint:
    """Upgrade a raw fingerprint dict to ``target`` schema version.

    In v1.0.0 only one schema version exists, so this validates the input
    version and constructs a :class:`Fingerprint` directly. Future versions
    will dispatch on the source version and apply a chain of upgrade
    functions registered here.

    ``raw`` is annotated as ``dict[str, Any]`` rather than a tighter
    JSON-value type because it is the deliberate boundary between
    untrusted external bytes and validated MirrorML state; downstream code
    operates on the :class:`Fingerprint` instance.

    Raises :class:`~mirrorml.exceptions.FingerprintVersionError` for
    documents missing or declaring an unsupported version.
    """

    version = raw.get("schema_version")
    if not isinstance(version, str):
        raise FingerprintVersionError(
            "fingerprint document is missing a 'schema_version' field "
            "or it is not a string; the document does not look like a "
            "MirrorML fingerprint."
        )

    if version < MIN_SUPPORTED_SCHEMA_VERSION:
        raise FingerprintVersionError(
            f"fingerprint schema version {version!r} is older than the "
            f"minimum supported version {MIN_SUPPORTED_SCHEMA_VERSION!r}; "
            f"no migration path is currently registered. Re-generate the "
            f"fingerprint with the current MirrorML release."
        )

    if version != target:
        raise FingerprintVersionError(
            f"fingerprint schema version {version!r} cannot be migrated to "
            f"target {target!r}; no migration path is registered. Either "
            f"target the source version explicitly or upgrade MirrorML."
        )

    return Fingerprint.model_validate(raw)
