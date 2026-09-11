"""Acceptance criteria for a stationary, unsupported two-hand grasp."""

import math


def evaluate_hold(samples, release_time, requested_duration):
    free = [s for s in samples if s["t"] >= release_time]
    reasons = []
    fields = (
        "t",
        "core_position_error_m",
        "core_rotation_error_deg",
        "joint_torque_max_nm",
    )
    if any(not math.isfinite(s[k]) for s in samples for k in fields):
        reasons.append("non-finite measurements")
    if any(not 0 < b["t"] - a["t"] <= 0.021 for a, b in zip(free, free[1:])):
        reasons.append("missing or non-monotonic samples")
    if not free or free[-1]["t"] - release_time < 5 - 0.021:
        reasons.append("less than five seconds of unsupported holding")
    if not samples or samples[-1]["t"] < requested_duration - 0.021:
        reasons.append("recording ended early")
    if any(s["core_position_error_m"] > 0.01 for s in free):
        reasons.append("cube drift exceeded 10 mm")
    if any(s["core_rotation_error_deg"] > 10 for s in free):
        reasons.append("cube rotation exceeded 10 degrees")
    if any(not s["anchors_connected"] for s in free):
        reasons.append("cube joint anchors separated")
    if any(s["joint_torque_max_nm"] > 1.45 for s in free):
        reasons.append("finger actuator approached its torque limit")
    fractions = {
        side: sum(sum(s["contact_loads_n"][side]) > 0.25 for s in free)
        / max(1, len(free))
        for side in ("left", "right")
    }
    if any(f < 0.95 for f in fractions.values()):
        reasons.append("both hands did not maintain cube contact")
    return dict(
        passed=not reasons,
        reasons=reasons,
        contact_fraction=fractions,
        max_position_error_m=max(
            (s["core_position_error_m"] for s in free), default=None
        ),
        max_rotation_error_deg=max(
            (s["core_rotation_error_deg"] for s in free), default=None
        ),
    )
