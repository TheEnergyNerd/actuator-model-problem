"""Measured body-state export for native Factory assets, before episode resets."""

import json
from pathlib import Path
import numpy as np


class FactoryRecording:
    def __init__(self, output, env):
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics, Gf

        self.output = Path(output) / "replay"
        self.output.mkdir()
        self.env = env
        self.assets = [env._robot, env._fixed_asset, env._held_asset]
        # GearMesh has two additional physical flanking gears.
        for name in ("_small_gear_asset", "_large_gear_asset"):
            asset = getattr(env, name, None)
            if asset is not None:
                self.assets.append(asset)
        self.frames, self.samples = [], []
        stage = omni.usd.get_context().get_stage()
        meshes, names = [], []
        for asset in self.assets:
            root = stage.GetPrimAtPath(asset.cfg.prim_path.replace("env_.*", "env_0"))
            bodies = {
                p.GetName(): p
                for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                if p.HasAPI(UsdPhysics.RigidBodyAPI)
            }
            for name in asset.body_names:
                body = bodies[name]
                world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
                rigid = Gf.Matrix4d(1)
                rigid.SetRotate(Gf.Transform(world).GetRotation())
                rigid.SetTranslateOnly(world.ExtractTranslation())
                index = len(names)
                names.append(str(body.GetPath()))
                for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                    if (
                        not prim.IsA(UsdGeom.Mesh)
                        or "collision" in str(prim.GetPath()).lower()
                    ):
                        continue
                    if UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
                        continue
                    mesh = UsdGeom.Mesh(prim)
                    points, counts, indices = (
                        mesh.GetPointsAttr().Get(),
                        mesh.GetFaceVertexCountsAttr().Get(),
                        mesh.GetFaceVertexIndicesAttr().Get(),
                    )
                    if not points or not counts:
                        continue
                    transform = (
                        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                            Usd.TimeCode.Default()
                        )
                        * rigid.GetInverse()
                    )
                    vertices = [
                        float(v)
                        for pt in points
                        for v in transform.Transform(Gf.Vec3d(pt))
                    ]
                    triangles, start = [], 0
                    for count in counts:
                        for j in range(1, count - 1):
                            triangles.extend(
                                [
                                    int(indices[start]),
                                    int(indices[start + j]),
                                    int(indices[start + j + 1]),
                                ]
                            )
                        start += count
                    color = (
                        [0.72, 0.76, 0.69]
                        if asset is env._robot
                        else (
                            [0.64, 0.45, 0.2]
                            if asset is env._held_asset
                            else [0.3, 0.37, 0.4]
                        )
                    )
                    meshes.append(
                        dict(
                            body=index,
                            vertices=vertices,
                            indices=triangles,
                            color=color,
                        )
                    )
        if not meshes:
            raise ValueError("No native assembly geometry")
        self.scene = dict(
            kind="assembly",
            bodies=names,
            meshes=meshes,
            static_meshes=[],
            cube_size=0,
            table=None,
            follow_body=len(env._robot.body_names),
            camera_offset=(
                [0.22, -0.28, 0.22] if len(self.assets) > 3 else [0.14, -0.18, 0.12]
            ),
        )
        (self.output / "scene.json").write_text(
            json.dumps(self.scene, separators=(",", ":"))
        )

    def capture(self, row):
        import torch

        state = (
            torch.cat(
                [
                    torch.cat((asset.data.body_pos_w[0], asset.data.body_quat_w[0]), -1)
                    for asset in self.assets
                ]
            )
            .detach()
            .cpu()
            .numpy()
        )
        if not np.isfinite(state).all():
            raise ValueError("Nonfinite assembly pose")
        self.frames.append(state)
        self.samples.append(
            dict(
                t=(len(self.frames) - 1) * self.env.step_dt,
                phase=(
                    "Side push"
                    if row.get("push_force_n", 0) > 0
                    else (
                        "Insertion achieved"
                        if row["success"][0]
                        else "Align and insert"
                    )
                ),
                push_force_n=row.get("push_force_n", 0),
                cube=state[self.scene["follow_body"], :3].tolist(),
                tips=[],
                grip=[],
                failures=0,
                torque=row["peak_joint_torque_nm"][0],
                keypoint_error_m=row["keypoint_error_m"][0],
                success=row["success"][0],
            )
        )

    def finish(self, evaluation):
        states = np.asarray(self.frames, dtype="<f4")
        states.tofile(self.output / "poses.bin")
        (self.output / "telemetry.json").write_text(
            json.dumps(self.samples, allow_nan=False)
        )
        trial = next(
            t
            for t in evaluation["trials"]
            if t["environment"] == 0 and t["episode"] == 0
        )
        result = dict(
            frames=len(states),
            bodies=states.shape[1],
            fps=1 / self.env.step_dt,
            duration=(len(states) - 1) * self.env.step_dt,
            physics_hz=1 / self.env.cfg.sim.dt,
            task_success=trial["sustained_final_success"],
            task=evaluation["task"],
            perturbation=evaluation.get("perturbation"),
            checkpoint_sha256=evaluation["checkpoint_sha256"],
            seed=evaluation["seed"],
            environment_index=0,
            episode_index=0,
            selection="First environment, first episode; selected before evaluation",
            reset_policy="Native prepared grasp; terminal body state captured before automatic reset",
            source="Measured native Isaac Lab Factory policy rollout",
        )
        (self.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
