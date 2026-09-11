"""Independently audit browser-format body poses against the requested cube moves."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cube_state import decode, state_after
from validation import measure_cube, stable_match


def audit(path):
    result = json.loads((path / "result.json").read_text())
    samples = json.loads((path / "telemetry.json").read_text())
    scene_path = path / "scene.json"
    if not scene_path.exists():
        scene_path = path.parent / "scene.json"
    scene = json.loads(scene_path.read_text())
    assert scene["bodies"][:27] == ["core"] + [f"cubie_{i:02d}" for i in range(26)]
    states = np.fromfile(path / "poses.bin", dtype="<f4").reshape(
        result["frames"], result["bodies"], 7
    )
    assert len(samples) == len(states) and len(scene["bodies"]) == result["bodies"]
    assert np.isfinite(states).all()
    assert np.max(np.abs(np.linalg.norm(states[:, :, 3:7], axis=-1) - 1)) < 1e-4
    times = np.array([row["t"] for row in samples])
    np.testing.assert_allclose(np.diff(times), 1 / result["fps"], atol=1e-7)
    assert abs(times[-1] - result["duration"]) < 1e-7
    measured = [
        dict(t=row["t"], **measure_cube(s[:27])) for row, s in zip(samples, states)
    ]
    cfg = result["configuration"]
    moves = [
        q
        for m in cfg["moves"].split()
        for q in ([m[0], m[0]] if m.endswith("2") else [m])
    ]
    completed = 0
    checkpoints = []
    for index, move in enumerate(moves):
        indices = [i for i, row in enumerate(samples) if row["move_index"] == index]
        if not indices:
            break
        end = indices[-1]
        expected = decode(
            state_after(cfg["scramble"] + " " + " ".join(moves[: index + 1]))
        )["facelets"]
        passed = stable_match(measured[: end + 1], expected)
        checkpoints.append(
            dict(
                quarter_turn=index + 1,
                move=move,
                end_time_s=samples[end]["t"],
                passed=passed,
            )
        )
        if not passed:
            break
        completed += 1
    initial_matches = (
        measured[0]["decoded"]["facelets"]
        == decode(state_after(cfg["scramble"]))["facelets"]
    )
    return dict(
        initial_state_matches=initial_matches,
        completed_quarter_turns=completed,
        requested_quarter_turns=len(moves),
        passed=bool(initial_matches and moves and completed == len(moves)),
        final_solved=measured[-1]["decoded"]["solved"],
        checkpoints=checkpoints,
        max_anchor_error_m=max(row["max_anchor_error_m"] for row in measured),
        thresholds=dict(sticker_degrees=3, anchor_metres=0.0005, hold_seconds=0.4),
        source_sha256={
            name: hashlib.sha256((path / name).read_bytes()).hexdigest()
            for name in ("poses.bin", "telemetry.json", "result.json")
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.recording)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
