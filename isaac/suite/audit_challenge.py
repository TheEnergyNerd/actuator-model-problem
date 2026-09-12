"""Audit recorded challenge geometry, route, reset flags and applied disturbance."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from challenge_course import FINISH_X, corridor_limit


def audit(root):
    result = json.loads((root / "result.json").read_text())
    rows = json.loads((root / "telemetry.json").read_text())
    poses = np.fromfile(root / "poses.bin", dtype="<f4").reshape(
        result["frames"], result["bodies"], 7
    )
    assert len(rows) == len(poses) > 1
    assert np.isfinite(poses).all()
    assert np.max(np.abs(np.linalg.norm(poses[:, :, 3:], axis=-1) - 1)) < 1e-3
    np.testing.assert_allclose(
        [r["t"] for r in rows], np.arange(len(rows)) / result["fps"], atol=1e-5
    )
    xyz = poses[:, 0, :3]
    lateral_ok = all(abs(float(y)) <= corridor_limit(float(x)) for x, y, z in xyz)
    # Clearance guards prevent counting passage on the lower pit floor as crossing.
    gap = xyz[(xyz[:, 0] >= 4.6) & (xyz[:, 0] <= 4.82)]
    bridge = xyz[(xyz[:, 0] >= 8.2) & (xyz[:, 0] <= 10.2)]
    clearance_ok = all(z >= 0.55 for x, y, z in gap) and all(
        z >= 0.4 for x, y, z in bridge
    )
    reset_free = result["episode_resets"] == 0
    finish = bool(xyz[-1, 0] >= FINISH_X)
    tail = xyz[-(round(result["fps"]) + 1) :, :2]
    final_speed = float(
        np.linalg.norm(np.diff(tail, axis=0), axis=1).max() * result["fps"]
    )
    force = np.array([r.get("push_force_body_y_N", 0) for r in rows])
    assert np.isin(force, [0, 70]).all()
    impulse = float(force.sum() / result["fps"])
    if result["task_success"]:
        assert (
            lateral_ok and clearance_ok and reset_free and finish and final_speed < 0.11
        )
        assert len(gap) and len(bridge) and 16 <= impulse <= 20
        assert result["passed_stage_ids"] == sorted(
            ["stairs", "gap", "bridge", "rubble", "slope", "push", "finish"]
        )
    return dict(
        finite_states=True,
        normalized_quaternions=True,
        regular_timestamps=True,
        route_corridor_ok=lateral_ok,
        clearance_ok=clearance_ok,
        reset_free=reset_free,
        reached_finish=finish,
        final_second_max_pose_speed_m_s=final_speed,
        applied_push_impulse_N_s=impulse,
        minimum_gap_base_height_m=float(gap[:, 2].min()) if len(gap) else None,
        minimum_bridge_base_height_m=float(bridge[:, 2].min()) if len(bridge) else None,
        clearance_note="Additional post-run audit: base height at least 0.55 m over the gap and 0.4 m over the bridge.",
        hashes={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ["poses.bin", "telemetry.json", "result.json"]
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    (args.root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
