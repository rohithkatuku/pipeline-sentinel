"""Schema drift detection — catch column adds, drops, type changes."""

import hashlib
import json
from dataclasses import dataclass


@dataclass
class SchemaDiff:
    added_columns: list[str]
    removed_columns: list[str]
    type_changes: list[dict]  # {"column": str, "old_type": str, "new_type": str}
    has_drift: bool
    severity: str
    summary: str


def schema_hash(columns: list[dict]) -> str:
    """Deterministic hash of a schema for quick comparison."""
    normalized = sorted(columns, key=lambda c: c["name"])
    return hashlib.sha256(json.dumps(normalized).encode()).hexdigest()[:16]


def detect_drift(previous: list[dict], current: list[dict]) -> SchemaDiff:
    """Compare two schema snapshots and report changes.

    Args:
        previous: List of {"name": str, "type": str} dicts.
        current: Same format, new schema.
    """
    prev_map = {c["name"]: c.get("type", "unknown") for c in previous}
    curr_map = {c["name"]: c.get("type", "unknown") for c in current}

    prev_names = set(prev_map.keys())
    curr_names = set(curr_map.keys())

    added = sorted(curr_names - prev_names)
    removed = sorted(prev_names - curr_names)

    type_changes = []
    for col in prev_names & curr_names:
        if prev_map[col] != curr_map[col]:
            type_changes.append({
                "column": col,
                "old_type": prev_map[col],
                "new_type": curr_map[col]
            })

    has_drift = bool(added or removed or type_changes)

    # Severity logic
    if removed or type_changes:
        severity = "high"  # breaking changes
    elif added:
        severity = "medium"  # additive, usually safe
    else:
        severity = "low"

    # Build summary
    parts = []
    if added:
        parts.append(f"Added: {', '.join(added)}")
    if removed:
        parts.append(f"Removed: {', '.join(removed)}")
    if type_changes:
        tc_strs = [f"{c['column']} ({c['old_type']}→{c['new_type']})" for c in type_changes]
        parts.append(f"Type changed: {', '.join(tc_strs)}")

    summary = "; ".join(parts) if parts else "No schema changes detected."

    return SchemaDiff(
        added_columns=added,
        removed_columns=removed,
        type_changes=type_changes,
        has_drift=has_drift,
        severity=severity,
        summary=summary
    )
