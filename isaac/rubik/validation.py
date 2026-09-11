"""Validation from measured body poses, independent of wrist commands."""

import numpy as np
from scipy.spatial.transform import Rotation
from cube_state import decode


def measure_cube(states, anchor_tolerance_m=0.0005):
    """Core followed by 26 cubies; all joint anchors coincide at body origins."""
    states = np.asarray(states)
    if states.shape[0] != 27 or not np.isfinite(states).all():
        raise ValueError("Expected 27 finite measured cube body states")
    r = Rotation.from_quat(states[:, [4, 5, 6, 3]]).as_matrix()
    decoded = decode(r[0].T @ r[1:])
    error = float(np.linalg.norm(states[1:, :3] - states[0, :3], axis=1).max())
    return dict(
        decoded=decoded,
        max_anchor_error_m=error,
        anchors_connected=error <= anchor_tolerance_m,
    )


def stable_match(samples, expected, hold_seconds=0.4):
    """Require a continuous measured hold, not one fortunate aligned frame."""
    if not samples or not expected:
        return False
    end = samples[-1]["t"]
    start = end - hold_seconds
    prior = [i for i, sample in enumerate(samples) if sample["t"] <= start + 1e-8]
    if not prior:
        return False
    window = samples[prior[-1] :]
    return all(
        sample.get("anchors_connected", False)
        and sample["decoded"]["facelets"] == expected
        for sample in window
    )
