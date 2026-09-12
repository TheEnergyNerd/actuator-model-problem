"""Export native terrain and measured env-0 state, without synthesizing motion."""

import json
from pathlib import Path
import sys
import numpy as np


def export_recording(output, env, frames, rows, summary):
    from pxr import Usd, UsdGeom, Gf
    import omni.usd

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "precision" / "eval_scripts")
    )
    from locomotion_replay import LocomotionRecorder

    dest = output / "replay"
    dest.mkdir()
    LocomotionRecorder(dest, env, [])
    scene = json.loads((dest / "scene.json").read_text())
    for mesh in scene["meshes"]:
        mesh["color"] = [0.24, 0.29, 0.25]
    stage = omni.usd.get_context().get_stage()
    stage.GetRootLayer().Export(str(output / "scene.usda"))
    static = []
    lo = frames[:, 0, :, :2].min(axis=(0, 1)) - 3
    hi = frames[:, 0, :, :2].max(axis=(0, 1)) + 3
    terrain = stage.GetPrimAtPath(env.cfg.scene.terrain.prim_path)
    for prim in Usd.PrimRange(terrain, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        vertices = np.array([matrix.Transform(Gf.Vec3d(p)) for p in points])
        counts, indices = (
            mesh.GetFaceVertexCountsAttr().Get(),
            mesh.GetFaceVertexIndicesAttr().Get(),
        )
        faces, offset = [], 0
        for count in counts:
            for j in range(1, count - 1):
                faces.append(
                    [indices[offset], indices[offset + j], indices[offset + j + 1]]
                )
            offset += count
        faces = np.asarray(faces, dtype=np.int32)
        triangles = vertices[faces, :2]
        keep = np.all(triangles.max(axis=1) >= lo, axis=1) & np.all(
            triangles.min(axis=1) <= hi, axis=1
        )
        faces = faces[keep]
        if not len(faces):
            continue
        used, remap = np.unique(faces, return_inverse=True)
        static.append(
            dict(
                vertices=np.round(vertices[used], 6).ravel().tolist(),
                indices=remap.tolist(),
                color=(
                    list(mesh.GetDisplayColorAttr().Get()[0])
                    if mesh.GetDisplayColorAttr().Get()
                    else [0.55, 0.60, 0.52]
                ),
            )
        )
    if not static:
        raise ValueError("No native terrain geometry exported")
    scene["static_meshes"] = static
    if summary.get("course_layout") == "challenge":
        scene["overview_camera"] = dict(eye=[16, -27, 21], target=[9, 0, 0.3])
    (dest / "scene.json").write_text(
        json.dumps(scene, separators=(",", ":"), allow_nan=False)
    )
    states = frames[:, 0].astype("<f4")
    states.tofile(dest / "poses.bin")
    samples, resets, failures = [], 0, 0
    for i, row in enumerate(rows):
        failures += int(row["terminated"][0])
        resets += int(row["terminated"][0] or row["time_out"][0])
        samples.append(
            dict(
                t=i * env.step_dt,
                phase=(
                    "Episode reset"
                    if row["time_out"][0]
                    else row.get("phase", "Rough terrain")
                ),
                push_force_body_y_N=row.get("push_force_body_y_N", 0),
                peak_joint_rpm=row.get("peak_joint_rpm"),
                cube=states[i, 0, :3].tolist(),
                tips=[],
                grip=[],
                failures=resets,
                failure_terminations=failures,
                speed=row["velocity"][0][0],
                target_speed=row["command"][0][0],
                temperature=row.get("temperature_c"),
                torque=row.get("peak_joint_torque_nm"),
                requested_torque=None,
                current=row.get("peak_current_a"),
                saturation=None,
            )
        )
    (dest / "telemetry.json").write_text(json.dumps(samples, allow_nan=False))
    result = dict(
        frames=len(states),
        bodies=states.shape[1],
        fps=1 / env.step_dt,
        duration=(len(states) - 1) * env.step_dt,
        physics_hz=1 / env.cfg.sim.dt,
        source_first_timestamp_s=rows[0]["t"],
        seed=summary["seed"],
        checkpoint_sha256=summary["checkpoint_sha256"],
        failure_terminations=failures,
        episode_resets=resets,
        actuator_model=summary["actuator_model"],
        environment_index=0,
        task_success=summary["task_success"],
        terrain=summary["terrain"],
        design=summary.get("design"),
        course_layout=summary.get("course_layout"),
        stages=summary.get("stages", []),
        passed_stage_ids=summary.get("passed_stage_ids", []),
        push_start_s=summary.get("push_start_s"),
        navigation=summary.get("navigation"),
        source="Measured native Isaac Lab rough-terrain rollout",
    )
    (dest / "result.json").write_text(json.dumps(result, indent=2) + "\n")
