"""A fixed, reproducible PhysX course: stairs, platform, descent and blocks."""

from pathlib import Path
import numpy as np

FINISH_X = 12.5


def make_course(path: Path):
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    import trimesh

    shapes = []

    def box(size, center):
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_translation(center)
        shapes.append(mesh)

    box((20, 5, 0.1), (7, 0, -0.05))
    for i in range(5):
        height = 0.08 * (i + 1)
        box((0.35, 3, height), (2 + 0.35 * (i + 0.5), 0, height / 2))
    box((1.5, 3, 0.4), (4.5, 0, 0.2))
    for i in range(5):
        height = 0.08 * (5 - i)
        box((0.35, 3, height), (5.25 + 0.35 * (i + 0.5), 0, height / 2))
    for i in range(5):
        for j in range(5):
            height = [0.04, 0.08, 0.12][(i + 2 * j) % 3]
            box(
                (0.45, 0.45, height),
                (7.7 + 0.60 * i, -0.0 + 0.55 * (j - 2), height / 2),
            )
    joined = trimesh.util.concatenate(shapes)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = UsdGeom.Xform.Define(stage, "/Course")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Course/Terrain")
    mesh.CreatePointsAttr(np.asarray(joined.vertices, dtype=np.float32).tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(joined.faces))
    mesh.CreateFaceVertexIndicesAttr(joined.faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.55, 0.47)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()


def configure_course(cfg, path):
    from isaaclab.terrains import TerrainImporterCfg

    make_course(path)
    cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=str(path),
        env_spacing=25.0,
        physics_material=cfg.scene.terrain.physics_material,
    )
    cfg.curriculum.terrain_levels = None
    cfg.episode_length_s = 35
    command = cfg.commands.base_velocity
    command.heading_command = False
    command.rel_standing_envs = 0
    command.ranges.lin_vel_x = (0.8, 0.8)
    command.ranges.lin_vel_y = (0, 0)
    command.ranges.ang_vel_z = (0, 0)
    command.resampling_time_range = (40, 40)
    cfg.events.reset_base.params["pose_range"] = dict(x=(0, 0), y=(0, 0), yaw=(0, 0))
    cfg.events.reset_base.params["velocity_range"] = dict(
        x=(0, 0), y=(0, 0), z=(0, 0), roll=(0, 0), pitch=(0, 0), yaw=(0, 0)
    )
