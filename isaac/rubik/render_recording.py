"""Render measured Isaac body states; this script never evaluates or invents motion."""

import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--recording", type=Path, required=True)
p.add_argument("--fps", type=int, default=25)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app
import numpy as np, torch, imageio.v2 as imageio, json
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg


def main():
    output = a.recording / "video.mp4"
    if output.exists():
        raise FileExistsError(output)
    data = np.load(a.recording / "poses.npz")
    states = data["states"]
    names = list(data["names"])
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.001, device="cpu"))
    stage = sim.stage
    stage.GetRootLayer().subLayerPaths.append(
        str((a.recording / "scene.usda").resolve())
    )
    bodies = {}
    for prim in list(stage.Traverse()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
            if prim.GetName() in names:
                bodies[prim.GetName()] = prim
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.IsA(UsdPhysics.Joint):
            prim.SetActive(False)
    if set(names) != set(bodies):
        raise ValueError("Recorded body/scene mismatch")
    ops = []
    for name in names:
        prim = bodies[name]
        parent = (
            UsdGeom.Xformable(prim.GetParent())
            .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            .GetInverse()
        )
        x = UsdGeom.Xformable(prim)
        x.ClearXformOpOrder()
        op = x.AddTransformOp(opSuffix="replay")
        ops.append((op, parent))
    camera = Camera(
        CameraCfg(
            prim_path="/World/ReplayCamera",
            height=1080,
            width=1080,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24,
                focus_distance=0.45,
                horizontal_aperture=20,
                clipping_range=(0.01, 10),
            ),
        )
    )
    sim.reset()
    camera.set_world_poses_from_view(
        torch.tensor([[0.40, -0.50, 0.95]]), torch.tensor([[0, 0, 0.58]])
    )
    writer = imageio.get_writer(
        str(output), fps=a.fps, codec="libx264", quality=8, macro_block_size=1
    )
    stride = 50 // a.fps
    if 50 % a.fps:
        raise ValueError("Render FPS must divide the recorded 50 FPS")
    view_center = None
    view_distance = None
    view_direction = np.array([0.70, 0.20, 0.65])
    view_direction /= np.linalg.norm(view_direction)
    for index in range(0, len(states), stride):
        points = states[index, :, :3]
        lo, hi = points.min(0), points.max(0)
        center = (lo + hi) / 2
        distance = (np.linalg.norm(hi - lo) / 2 + 0.055) / np.sin(np.arctan(10 / 24))
        view_center = (
            center if view_center is None else view_center * 0.88 + center * 0.12
        )
        view_distance = (
            distance
            if view_distance is None
            else view_distance * 0.88 + distance * 0.12
        )
        camera.set_world_poses_from_view(
            torch.tensor(
                [view_center + view_direction * view_distance], dtype=torch.float
            ),
            torch.tensor([view_center], dtype=torch.float),
        )
        for values, (op, parent) in zip(states[index], ops):
            p, q = values[:3], values[3:7]
            world = Gf.Matrix4d(1)
            world.SetRotate(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))))
            world.SetTranslateOnly(Gf.Vec3d(*map(float, p)))
            op.Set(world * parent)
        sim.render()
        camera.update(1 / a.fps, force_recompute=True)
        writer.append_data(camera.data.output["rgb"][0, :, :, :3].cpu().numpy())
        if index % 500 == 0:
            print("RENDER", index, "/", len(states), flush=True)
    writer.close()
    (a.recording / "render.json").write_text(
        json.dumps(
            {
                "source": "poses.npz",
                "renderer": "Isaac Sim 4.5",
                "method": "Offline render of measured physics states; physics disabled during rendering",
                "camera": "Smooth framing of measured body bounds; no change to body motion",
                "video_fps": a.fps,
                "physics_replay_fps": 50,
            },
            indent=2,
        )
    )


main()
import threading, os

threading.Timer(5, lambda: os._exit(0)).start()
app.close()
