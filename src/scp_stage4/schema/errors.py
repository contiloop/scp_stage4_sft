from __future__ import annotations


class SchemaValidationError(ValueError):
    """Raised when a row does not match the expected schema contract."""

