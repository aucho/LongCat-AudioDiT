"""Cross-platform validation shared by API schemas and persistence code."""

from __future__ import annotations

import re

from pydantic import AfterValidator
from typing_extensions import Annotated


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def validate_resource_id(value: str, field_name: str = "id") -> str:
    """Return a portable resource id or raise ``ValueError``."""
    value = (value or "").strip()
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(
            f"{field_name} must contain only letters, numbers, '.', '_' or '-'"
        )
    if value.endswith(".") or ".." in value:
        raise ValueError(f"{field_name} must not contain '..' or end with '.'")
    if value.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        raise ValueError(f"{field_name} is a reserved file name")
    return value


def _validate_schema_resource_id(value: str) -> str:
    return validate_resource_id(value, "resource id")


ResourceId = Annotated[str, AfterValidator(_validate_schema_resource_id)]


__all__ = ["ResourceId", "validate_resource_id"]
