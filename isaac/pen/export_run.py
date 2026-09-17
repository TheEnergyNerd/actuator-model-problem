"""Score Isaac trajectories and export one preselected measured replay, without MuJoCo."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def export(run, reference, trial=0):
    sys.path.insert(0, str(reference / "pen/policy"))
    sys.path.insert(0, str(reference / "pen/sim"))
    from gate import check_trial
    from spec import FINGERS, JOINTS

    z = np.load(run / "trajectories.npz")
    body = np.load(run / "body_states.npz")
    meta = json.loads((run / "meta.json").read_text())
    provenance = json.loads((run / "reference.json").read_text())
    slices = {f: [i for i, j in enumerate(JOINTS) if f"_{f}_" in j] for f in FINGERS}
    results = []
    for i in range(meta["trials"]):
        rec = {
            k: z[k][i]
            for k in ("pen_pos", "pen_axis", "pen_angvel", "q", "qd", "finger_force")
        }
        rec.update(ref_pos=z["ref_pos"][i], heading0=z["heading0"][i])
        # Initial separation was not recorded upstream. Conservatively reuse the
        # first measurement and explicitly avoid claiming the upstream full gate.
        pen = 1000 * z["physx_penetration"][i]
        _, det = check_trial(rec, slices, z["q_lo"], z["q_hi"], np.r_[pen[0], pen])
        det.update(trial=i, seed=meta["seed0"] + i, unsettled=bool(z["unsettled"][i]))
        det["motion_passed"] = bool(
            det["turns_ok"]
            and det["timing_ok"]
            and det["hold_ok"]
            and not det["drop"]
            and det["limits_ok"]
            and not det["unsettled"]
        )
        det["native_contact_gate_passed"] = bool(
            det["motion_passed"]
            and det["penetration_ok"]
            and det["participation_ok"]
            and det["all_fingers_ok"]
            and det["thumb_release_ok"]
        )
        results.append(det)
    evaluation = dict(
        trials=results,
        motion_successes=sum(r["motion_passed"] for r in results),
        native_contact_successes=sum(r["native_contact_gate_passed"] for r in results),
        method="Upstream motion thresholds; PhysX contact separations replace independent MuJoCo hull replay. Initial penetration is not measured. Not an upstream gate reproduction.",
    )
    root = run / "replay"
    (root / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
    states = body["poses"][trial].astype("<f4")
    if len(states) != meta["steps"] or not np.isfinite(states).all():
        raise ValueError("Body recording is incomplete or nonfinite")
    if not np.allclose(np.linalg.norm(states[:, :, 3:], axis=-1), 1, atol=1e-3):
        raise ValueError("Invalid recorded quaternion")
    if not np.allclose(states[:, -1, :3], z["pen_pos"][trial], atol=1e-5):
        raise ValueError("Recorded pen pose does not match evaluated trajectory")
    states.tofile(root / "poses.bin")
    turns = (
        np.unwrap(np.arctan2(z["pen_axis"][trial, :, 1], z["pen_axis"][trial, :, 0]))
        - z["heading0"][trial]
    ) / (2 * np.pi)
    samples = []
    for k, turn in enumerate(turns):
        samples.append(
            dict(
                t=k * meta["control_dt"],
                simulation_time=(k + 1) * meta["control_dt"],
                phase="Hold" if turn >= 3 else f"Rotation {min(3,max(1,int(turn)+1))}",
                turns=float(turn),
                cube=states[k, -1, :3].tolist(),
                tips=[],
                grip=[],
                torque=float(abs(body["torque"][trial, k]).max()),
                finger_forces=z["finger_force"][trial, k].tolist(),
                angular_speed_deg_s=float(
                    np.degrees(abs(z["pen_angvel"][trial, k, 2]))
                ),
                penetration_mm=float(1000 * z["physx_penetration"][trial, k]),
            )
        )
    (root / "telemetry.json").write_text(json.dumps(samples, separators=(",", ":")))
    result = dict(
        frames=len(states),
        bodies=states.shape[1],
        fps=60,
        physics_hz=240,
        duration=(len(states) - 1) / 60,
        source_time_offset_s=1 / 60,
        trial=trial,
        seed=meta["seed0"] + trial,
        task_success=results[trial]["motion_passed"],
        outcome=results[trial],
        provenance=provenance,
        actuation="Native Sharpa implicit position control; optional torque-ceiling or hand-mass intervention",
        intervention=(
            json.loads((run / "intervention.json").read_text())
            if (run / "intervention.json").exists()
            else {"effort_scale": 1.0, "hand_mass_scale": 1.0}
        ),
        poses_sha256=hashlib.sha256((root / "poses.bin").read_bytes()).hexdigest(),
    )
    (root / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {k: evaluation[k] for k in ["motion_successes", "native_contact_successes"]}
        )
    )
    print("Selected trial:", results[trial])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--trial", type=int, default=0)
    a = p.parse_args()
    export(a.run, a.reference, a.trial)
