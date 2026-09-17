"""Isaac render of exported native meshes at measured poses; no motion synthesis."""

import argparse
import json
import os
from pathlib import Path
import threading
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--recording", type=Path, required=True)
parser.add_argument("--fps", type=int, default=25)
AppLauncher.add_app_launcher_args(parser)
a = parser.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app
import numpy as np
import torch
import imageio.v2 as imageio
from pxr import UsdGeom, UsdShade, Sdf, Gf
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg


def main():
    root = a.recording / "replay"
    output = root / "video.mp4"
    if output.exists():
        raise FileExistsError(output)
    result = json.loads((root / "result.json").read_text())
    scene = json.loads((root / "scene.json").read_text())
    states = np.fromfile(root / "poses.bin", dtype="<f4").reshape(
        result["frames"], result["bodies"], 7
    )
    if result["fps"] % a.fps or a.fps <= 0:
        raise ValueError("Video rate must divide the recorded rate")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.02, device="cpu"))
    stage = sim.stage
    light = sim_utils.DomeLightCfg(
        intensity=180 if scene.get("kind") in ("assembly", "hand") else 700,
        color=(0.85, 0.90, 1),
    )
    light.func("/World/Light", light)
    sun = sim_utils.DistantLightCfg(
        intensity=300 if scene.get("kind") in ("assembly", "hand") else 1200,
        color=(1, 0.96, 0.85),
    )
    sun.func("/World/Sun", sun, orientation=(0.92388, 0.38268, 0, 0))
    ops = []
    for i in range(result["bodies"]):
        body = UsdGeom.Xform.Define(stage, f"/World/Body_{i}")
        ops.append(body.AddTransformOp())
    for i, mesh in enumerate(
        scene["meshes"] + [dict(m, body=-1) for m in scene["static_meshes"]]
    ):
        parent = "/World/Terrain" if mesh["body"] < 0 else f"/World/Body_{mesh['body']}"
        primitive = UsdGeom.Mesh.Define(stage, f"{parent}/Mesh_{i}")
        primitive.CreatePointsAttr(np.asarray(mesh["vertices"]).reshape(-1, 3).tolist())
        primitive.CreateFaceVertexCountsAttr([3] * (len(mesh["indices"]) // 3))
        primitive.CreateFaceVertexIndicesAttr(mesh["indices"])
        primitive.CreateSubdivisionSchemeAttr("none")
        primitive.CreateDisplayColorAttr([Gf.Vec3f(*mesh["color"])])
        material = UsdShade.Material.Define(stage, f"/World/Materials/Material_{i}")
        shader = UsdShade.Shader.Define(stage, f"/World/Materials/Material_{i}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shade = (
            mesh["color"]
            if scene.get("kind") in ("assembly", "hand")
            else [0.24, 0.29, 0.25] if mesh["body"] >= 0 else mesh["color"]
        )
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*shade)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        UsdShade.MaterialBindingAPI.Apply(primitive.GetPrim()).Bind(material)
    camera = Camera(
        CameraCfg(
            prim_path="/World/Camera",
            height=1080,
            width=1080,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24, horizontal_aperture=20, clipping_range=(0.05, 100)
            ),
        )
    )
    sim.reset()
    writer = imageio.get_writer(
        str(output), fps=a.fps, codec="libx264", quality=8, macro_block_size=1
    )
    stride = int(result["fps"] / a.fps)
    offset = (
        np.array(scene.get("camera_offset", [0.14, -0.18, 0.12]))
        if scene.get("kind") in ("assembly", "hand")
        else np.array([1.8, -2.4, 1.6])
    )
    try:
        for index in range(0, len(states), stride):
            for values, op in zip(states[index], ops):
                matrix = Gf.Matrix4d(1)
                matrix.SetRotate(
                    Gf.Quatd(float(values[3]), Gf.Vec3d(*map(float, values[4:7])))
                )
                matrix.SetTranslateOnly(Gf.Vec3d(*map(float, values[:3])))
                op.Set(matrix)
            center = states[index, scene.get("follow_body", 0), :3].astype(float)
            camera.set_world_poses_from_view(
                torch.tensor(np.array([center + offset]), dtype=torch.float32),
                torch.tensor(np.array([center]), dtype=torch.float32),
            )
            for _ in range(8 if index == 0 else 1):
                sim.render()
            camera.update(1 / a.fps, force_recompute=True)
            writer.append_data(camera.data.output["rgb"][0, :, :, :3].cpu().numpy())
            if index % 100 == 0:
                print("RENDER", index, "/", len(states), flush=True)
    finally:
        writer.close()
    (root / "render.json").write_text(
        json.dumps(
            dict(
                renderer="Isaac Sim",
                method="Native exported mesh geometry at measured body poses; display colors and studio lighting; physics disabled during rendering",
                source="poses.bin",
                video_fps=a.fps,
                replay_fps=result["fps"],
                video_frames=len(range(0, len(states), stride)),
            ),
            indent=2,
        )
        + "\n"
    )


code = 1
try:
    main()
    code = 0
finally:
    timer = threading.Timer(5, lambda: os._exit(code))
    timer.daemon = True
    timer.start()
    app.close()
