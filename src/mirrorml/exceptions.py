"""Custom exceptions raised by MirrorML's public API.

All public-API errors derive from :class:`MirrorMLError`. Messages must be
actionable in the form "X failed because Y, try Z" — see ``CLAUDE.md`` § Error
handling.
"""

from __future__ import annotations


class MirrorMLError(Exception):
    """Base class for every exception raised by MirrorML.

    Catch this to handle any MirrorML-originated failure; catch a subclass to
    handle a specific failure mode.
    """


class UnsupportedOperationError(MirrorMLError):
    """A tracer encountered an operation it does not yet support.

    The message names the offending operation (e.g. ``pandas.DataFrame.melt``)
    and links to the issue tracker so users can either request support or
    rewrite the pipeline using a supported equivalent.
    """


class FingerprintVersionError(MirrorMLError):
    """A fingerprint document declares a schema version this build cannot load.

    Cross-version loads never happen silently: callers must route old documents
    through :func:`mirrorml.fingerprint.migrate.migrate` to upgrade them to the
    current schema version.
    """


class CanonicalizationError(MirrorMLError):
    """The canonicalizer could not produce a deterministic encoding.

    Usually indicates a malformed operation graph (cycle, missing dependency,
    duplicate ``op_id``) or a programming error in a tracer. The message
    points at the offending operation so the responsible tracer can be fixed.
    """
