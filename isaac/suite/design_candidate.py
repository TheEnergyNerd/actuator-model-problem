"""Validate downloadable model candidates before starting Isaac Sim."""

import json
import math
import re
from pathlib import Path

RANGES = {
    "kv_factor": (0.5, 1.5),
    "gear_factor": (0.5, 1.5),
    "peak_current_factor": (0.5, 1.5),
    "added_mass_factor": (0, 2),
}


def load_candidate(path):
    path = Path(path)
    if path.stat().st_size > 65536:
        raise ValueError("Candidate must be smaller than 64 KiB")
    data = json.loads(path.read_text())
    if (
        not isinstance(data, dict)
        or set(data) != {"schema", "design"}
        or data["schema"] != "atlas-course-design-v1"
    ):
        raise ValueError("Expected atlas-course-design-v1 schema")
    design = data["design"]
    if not isinstance(design, dict) or set(design) != {"name", "axis", *RANGES}:
        raise ValueError("Candidate fields do not match the course contract")
    if not isinstance(design["name"], str) or not re.fullmatch(
        "[a-z][a-z0-9_]{0,63}", design["name"]
    ):
        raise ValueError("Invalid design name")
    if not isinstance(design["axis"], str) or len(design["axis"]) > 160:
        raise ValueError("Invalid design description")
    for key, (lo, hi) in RANGES.items():
        value = design[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not lo <= value <= hi
        ):
            raise ValueError(f"{key} must be finite and in [{lo}, {hi}]")
    return design
