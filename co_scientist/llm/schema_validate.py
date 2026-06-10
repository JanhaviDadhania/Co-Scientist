"""Dependency-free JSON-schema payload validation for forced tool calls.

Covers the subset of JSON Schema our tool input_schemas actually use:
object properties, required, string/integer/number/boolean/array/object
types, enum membership, and required keys of array-of-object items.
Returns human-readable error strings that are fed back to the model on
retry, so messages should say exactly what to fix.
"""

from __future__ import annotations

from typing import Any


def validate_payload(schema: dict[str, Any] | None, payload: Any) -> list[str]:
    """Validate `payload` against `schema`. Returns [] when valid."""
    if not isinstance(payload, dict):
        return [f"payload must be a JSON object, got {type(payload).__name__}"]
    schema = schema or {}
    props: dict[str, Any] = schema.get("properties") or {}
    errors: list[str] = []

    for req in schema.get("required") or []:
        if req not in payload:
            errors.append(f"missing required field: `{req}`")
        elif payload[req] is None or payload[req] == "":
            errors.append(f"required field `{req}` is empty")

    for key, value in payload.items():
        spec = props.get(key)
        if spec is None:
            continue  # tolerate extra fields
        errors.extend(_check(key, value, spec))

    return errors


def _check(path: str, value: Any, spec: dict[str, Any]) -> list[str]:
    t = spec.get("type")
    enum = spec.get("enum")
    errors: list[str] = []

    if value is None:
        return errors  # missing-ness is handled by `required`

    if enum is not None and value not in enum:
        errors.append(f"`{path}` must be one of {enum}, got {value!r}")

    if t == "string":
        if not isinstance(value, str):
            errors.append(f"`{path}` must be a string, got {type(value).__name__}")
    elif t == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"`{path}` must be an integer, got {type(value).__name__}")
    elif t == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"`{path}` must be a number, got {type(value).__name__}")
    elif t == "boolean":
        if not isinstance(value, bool):
            errors.append(f"`{path}` must be a boolean, got {type(value).__name__}")
    elif t == "object":
        if not isinstance(value, dict):
            errors.append(f"`{path}` must be an object, got {type(value).__name__}")
    elif t == "array":
        if not isinstance(value, list):
            errors.append(f"`{path}` must be an array, got {type(value).__name__}")
            return errors
        items_spec = spec.get("items") or {}
        for i, item in enumerate(value):
            if items_spec.get("type") == "object":
                if not isinstance(item, dict):
                    errors.append(f"`{path}[{i}]` must be an object, got {type(item).__name__}")
                    continue
                for req in items_spec.get("required") or []:
                    if req not in item or item[req] in (None, ""):
                        errors.append(f"`{path}[{i}]` is missing required key `{req}`")
                inner_props = items_spec.get("properties") or {}
                for k, v in item.items():
                    if k in inner_props:
                        errors.extend(_check(f"{path}[{i}].{k}", v, inner_props[k]))
            elif items_spec.get("type") == "string" and not isinstance(item, str):
                errors.append(f"`{path}[{i}]` must be a string, got {type(item).__name__}")
    return errors
