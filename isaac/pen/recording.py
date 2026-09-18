"""Measured body-state export for native Factory assets, before episode resets."""

import json
from pathlib import Path
import numpy as np


class PenRecording:
    def __init__(self, output, env):
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics, Gf

        self.output = Path(output) / "replay"
        self.output.mkdir()
        self.env = env
        self.assets = [env.robot, env.pen]
        self.frames, self.samples = [], []
        self.motor_samples = []
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
                        if asset is env.robot
                        else (
                            [0.64, 0.45, 0.2] if asset is env.pen else [0.3, 0.37, 0.4]
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
        # The PhysX pen is an analytic capsule; tessellate the same dimensions for display.
        from spec import PEN
        import math

        vertices, indices = [], []
        rings, sides = 24, 32
        half = PEN["length"] / 2 - PEN["radius"]
        for r in range(rings + 1):
            theta = math.pi * r / rings
            x = PEN["radius"] * math.cos(theta) + (half if r <= rings // 2 else -half)
            radius = PEN["radius"] * math.sin(theta)
            for k in range(sides):
                phi = 2 * math.pi * k / sides
                vertices.extend([x, radius * math.cos(phi), radius * math.sin(phi)])
        for r in range(rings):
            for k in range(sides):
                a = r * sides + k
                b = r * sides + (k + 1) % sides
                c = a + sides
                d = b + sides
                indices.extend([a, b, c, b, d, c])
        meshes.append(
            dict(
                body=len(names) - 1,
                vertices=vertices,
                indices=indices,
                color=[0.08, 0.40, 0.72],
            )
        )
        if not meshes:
            raise ValueError("No native assembly geometry")
        self.scene = dict(
            kind="hand",
            bodies=names,
            meshes=meshes,
            static_meshes=[],
            cube_size=0,
            table=None,
            follow_body=len(env.robot.body_names),
            camera_offset=[0.18, -0.25, 0.18],
            overview_camera=dict(eye=[0.34, -0.30, 0.83], target=[0.14, 0, 0.56]),
        )
        (self.output / "scene.json").write_text(
            json.dumps(self.scene, separators=(",", ":"))
        )

    def capture(self):
        import torch

        state = (
            torch.cat(
                [
                    torch.cat((a.data.body_pos_w, a.data.body_quat_w), -1)
                    for a in self.assets
                ],
                1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        state[:, :, :3] -= self.env.scene.env_origins.detach().cpu().numpy()[:, None, :]
        if not np.isfinite(state).all():
            raise ValueError("Nonfinite recorded body pose")
        self.frames.append(state)
        self.samples.append(self.env.robot.data.applied_torque.detach().cpu().numpy())

        if hasattr(self.env, "motor_state"):
            self.motor_samples.append(
                {
                    k: v.detach().cpu().numpy().copy()
                    for k, v in self.env.motor_state.items()
                }
            )

    def finish(self):
        if self.motor_samples:
            np.savez_compressed(
                self.output.parent / "motor_traces.npz",
                **{
                    k: np.stack([s[k] for s in self.motor_samples], 1)
                    for k in self.motor_samples[0]
                }
            )
        np.savez_compressed(
            self.output.parent / "body_states.npz",
            poses=np.stack(self.frames, 1),
            torque=np.stack(self.samples, 1),
        )
