"""Measured challenge course: real gap, raised crossing, rubble, slope and push."""

from pathlib import Path
import numpy as np

FINISH_X = 19.0
STAGES = [
    dict(id="stairs", name="Climb the staircase", start=2.0, end=3.6),
    dict(id="gap", name="Cross the gap", start=4.6, end=4.82),
    dict(id="bridge", name="Balance on the raised crossing", start=8.2, end=10.2),
    dict(id="rubble", name="Pick through irregular rubble", start=10.6, end=14.0),
    dict(id="slope", name="Traverse the side slope", start=14.0, end=16.0),
    dict(id="push", name="Recover from a sideways push", start=17.0, end=18.0),
    dict(id="finish", name="Stop inside the finish zone", start=19.0, end=19.6),
]


def stage_at(x):
    return next(
        (s["name"] for s in STAGES if s["start"] <= x <= s["end"]),
        "Approach the next obstacle",
    )


def corridor_limit(x):
    return 0.35 if 8.2 <= x <= 10.2 else 1.0 if 2 <= x <= 18 else 2.0


def make_challenge(path: Path):
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    import trimesh

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = UsdGeom.Xform.Define(stage, "/Course")
    stage.SetDefaultPrim(root.GetPrim())
    zones = {}

    def box(zone, size, center, roll=0, yaw=0):
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_transform(trimesh.transformations.euler_matrix(roll, 0, yaw))
        mesh.apply_translation(center)
        zones.setdefault(zone, []).append(mesh)

    box("PitFloor", (24, 6, 0.1), (9, 0, -0.40))
    box("Approach", (4, 4, 0.1), (0, 0, -0.05))
    for i in range(4):
        h = 0.1 * (i + 1)
        box("Stairs", (0.4, 2.4, h), (2 + 0.4 * (i + 0.5), 0, h / 2))
    box("GapTakeoff", (1, 2.4, 0.4), (4.1, 0, 0.2))
    box("GapLanding", (0.98, 2.4, 0.4), (5.31, 0, 0.2))
    for i in range(4):
        h = 0.1 * (4 - i)
        box("Descent", (0.4, 2.4, h), (5.8 + 0.4 * (i + 0.5), 0, h / 2))
    box("BridgeApproach", (0.8, 2.4, 0.1), (7.8, 0, -0.05))
    box("BridgeRamp", (0.35, 1.2, 0.09), (8.025, 0, 0.045))
    box("RaisedCrossing", (2, 1.05, 0.18), (9.2, 0, 0.09))
    box("BridgeExit", (0.4, 1.2, 0.09), (10.4, 0, 0.045))
    box("RubbleGround", (3.4, 2.4, 0.1), (12.3, 0, -0.05))
    rng = np.random.default_rng(7)
    for i in range(6):
        for j in range(4):
            h = float(rng.uniform(0.04, 0.14))
            box(
                "Rubble",
                (0.4, 0.38, h),
                (10.9 + i * 0.5, (j - 1.5) * 0.5, h / 2),
                yaw=float(rng.uniform(-0.3, 0.3)),
            )
    box("FinalGround", (6, 4, 0.1), (17, 0, -0.05))
    box("SideSlope", (2, 2.2, 0.14), (15, 0, 0.13), roll=np.deg2rad(8))
    colors = {
        "PitFloor": (0.22, 0.28, 0.27),
        "Approach": (0.48, 0.55, 0.49),
        "Stairs": (0.57, 0.61, 0.52),
        "GapTakeoff": (0.61, 0.48, 0.29),
        "GapLanding": (0.61, 0.48, 0.29),
        "RaisedCrossing": (0.28, 0.48, 0.45),
        "Rubble": (0.49, 0.43, 0.36),
        "SideSlope": (0.4, 0.5, 0.59),
        "FinalGround": (0.48, 0.55, 0.49),
    }
    # Isaac Lab 2.1 ray casting reads only the first mesh below its target.
    # Give the policy the complete surface, while keeping colored collision zones.
    joined = trimesh.util.concatenate(
        [part for parts in zones.values() for part in parts]
    )
    sensor = UsdGeom.Mesh.Define(stage, "/Course/SensorSurface")
    sensor.CreatePointsAttr(np.asarray(joined.vertices, dtype=np.float32).tolist())
    sensor.CreateFaceVertexCountsAttr([3] * len(joined.faces))
    sensor.CreateFaceVertexIndicesAttr(joined.faces.ravel().tolist())
    sensor.CreateVisibilityAttr("invisible")
    for name, parts in zones.items():
        joined = trimesh.util.concatenate(parts)
        mesh = UsdGeom.Mesh.Define(stage, "/Course/" + name)
        mesh.CreatePointsAttr(np.asarray(joined.vertices, dtype=np.float32).tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(joined.faces))
        mesh.CreateFaceVertexIndicesAttr(joined.faces.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*colors.get(name, (0.5, 0.55, 0.47)))])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()


def configure_challenge(cfg, path):
    from isaaclab.terrains import TerrainImporterCfg

    make_challenge(path)
    cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=str(path),
        env_spacing=30,
        physics_material=cfg.scene.terrain.physics_material,
    )
    cfg.curriculum.terrain_levels = None
    cfg.episode_length_s = 60
    command = cfg.commands.base_velocity
    command.heading_command = False
    command.rel_standing_envs = 0
    command.ranges.lin_vel_x = (0.8, 0.8)
    command.ranges.lin_vel_y = (0, 0)
    command.ranges.ang_vel_z = (0, 0)
    command.resampling_time_range = (65, 65)
    cfg.events.reset_base.params["pose_range"] = dict(x=(0, 0), y=(0, 0), yaw=(0, 0))
    cfg.events.reset_base.params["velocity_range"] = dict(
        x=(0, 0), y=(0, 0), z=(0, 0), roll=(0, 0), pitch=(0, 0), yaw=(0, 0)
    )
