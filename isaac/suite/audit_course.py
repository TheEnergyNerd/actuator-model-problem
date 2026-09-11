"""Audit exported course trajectories independently of the online finish flag."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(root):
    result = json.loads((root / "source-result.json").read_text())
    poses = np.fromfile(root / "poses.bin", dtype="<f4").reshape(
        result["frames"], result["bodies"], 7
    )
    samples = json.loads((root / "telemetry.json").read_text())
    assert np.isfinite(poses).all()
    assert np.allclose(np.linalg.norm(poses[:, :, 3:], axis=-1), 1, atol=1e-4)
    assert len(samples) == len(poses)
    assert np.allclose([s["t"] for s in samples], np.arange(len(poses)) / result["fps"])
    xy = poses[:, 0, :2]
    limits = np.where((xy[:, 0] >= 2) & (xy[:, 0] <= 10.7), 1.3, 2.3)
    outside = np.flatnonzero(abs(xy[:, 1]) > limits)
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * result["fps"]
    n = round(result["fps"])
    # Finite differences of float32 positions differ slightly from native velocity.
    stopped = bool(
        len(speed) >= n and np.all(speed[-n:] < 0.101) and np.all(xy[-n:, 0] >= 12.5)
    )
    passed = bool(not len(outside) and stopped and result["episode_resets"] == 0)
    return dict(
        passed=passed,
        corridor_passed=not bool(len(outside)),
        first_corridor_exit_s=(
            float(outside[0] / result["fps"]) if len(outside) else None
        ),
        stopped_hold_passed=stopped,
        reset_count=result["episode_resets"],
        online_finish_flag=result["task_success"],
        criterion="Root stays within ±1.3 m for x=2..10.7 m and ±2.3 m elsewhere; final second beyond x=12.5 m at <0.1 m/s, with 0.001 m/s finite-difference tolerance; no reset",
        limitations="Development seed 201; corridor criterion audited after these runs; no held-out reliability estimate",
        sha256={
            f: hashlib.sha256((root / f).read_bytes()).hexdigest()
            for f in ["source-result.json", "poses.bin", "telemetry.json", "video.mp4"]
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recording", type=Path)
    print(json.dumps(audit(p.parse_args().recording), indent=2))
