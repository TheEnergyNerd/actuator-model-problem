"""Free cube held by two native Wuji hands; physical U-turn diagnostic."""

import argparse, json
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--video", action="store_true")
p.add_argument("--closure", type=float, default=0.004)
p.add_argument("--support-closure", type=float, default=0.012)
p.add_argument("--fixture-warmup", type=float, default=2.0)
p.add_argument("--third-support", action="store_true")
p.add_argument("--prepared-support", action="store_true")
p.add_argument(
    "--support-finger",
    choices=("middle_finger", "ring_finger", "pinky"),
    default="middle_finger",
)
p.add_argument("--support-point", type=float, nargs=3, default=(0, 0.015, -0.025))
p.add_argument("--hold-only", action="store_true")
p.add_argument("--locked-cube", action="store_true")
p.add_argument("--duration", type=float, default=10.0)
p.add_argument("--contact-profile", choices=("stock", "rigid", "pad"), default="pad")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.prepared_support and not a.third_support:
    p.error("--prepared-support requires --third-support")
if a.locked_cube and not a.hold_only:
    p.error("A locked cube is only a grasp diagnostic; use --hold-only")
if a.duration < a.fixture_warmup + 5:
    p.error("Record at least five seconds after fixture release")
a.enable_cameras = a.video
app = AppLauncher(a).app
import numpy as np, torch
from scipy.spatial.transform import Rotation
from pxr import UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.assets import (
    RigidObjectCollection,
    RigidObjectCollectionCfg,
    RigidObjectCfg,
)
from isaaclab.utils.math import matrix_from_quat
from native_hand import MountedHand
from wuji_kinematics import Hand
from passive import local_torques
from validation import measure_cube, stable_match
from grasp_validation import evaluate_hold
from cube_scene import make_cube
from cube_state import decode, state_after
from export_replay import export_geometry


def main():
    a.output.mkdir(parents=True, exist_ok=False)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=0.001,
            device=a.device,
            render_interval=20,
            physx=sim_utils.PhysxCfg(
                solver_type=1,
                min_position_iteration_count=32,
                min_velocity_iteration_count=8,
            ),
        )
    )
    import carb

    carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    stage = sim.stage
    sim_utils.DomeLightCfg(intensity=1800).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=1800)
    )
    grasp_config = json.loads(
        (Path(__file__).parent / "reference_grasp.json").read_text()
    )
    reference_R = np.array(grasp_config["R"])
    Sl = np.array(grasp_config["Sl"])
    Sr = np.array(grasp_config["Sr"])
    names = make_cube(
        stage,
        fixture=a.fixture_warmup > 0,
        scramble="U'",
        orientation=reference_R,
        locked=a.locked_cube,
    )
    cube = RigidObjectCollection(
        RigidObjectCollectionCfg(
            rigid_objects={
                n: RigidObjectCfg(prim_path="/World/Cube/" + n, spawn=None)
                for n in names
            }
        )
    )
    support = {}
    if a.third_support:
        support, error = Hand("left", 0).support_finger(
            a.support_finger,
            reference_R @ np.array(a.support_point),
            reference_R @ Sl,
        )
        if error > 0.0001:
            raise ValueError("Support fingertip IK cannot reach the requested point")
    left = MountedHand(
        stage,
        "left",
        0.0,
        reference_R @ Sl,
        min(0.003, a.support_closure),
        contact_profile=a.contact_profile,
        initial_joints=support if a.prepared_support else None,
    )
    right_frame = reference_R @ Sr
    right_origin = np.array([0, 0, 0.5]) + 0.01905 * (
        reference_R[:, 2] - right_frame[:, 2]
    )
    right = MountedHand(
        stage,
        "right",
        0.01905,
        right_frame,
        min(0.003, a.closure),
        cube_pos=right_origin,
        contact_profile=a.contact_profile,
    )
    hands = [left, right]
    left_goal = left.q | left.hand.grasp(a.support_closure)[0]
    if a.third_support:
        left_goal.update(support)
    right_goal = right.q | right.hand.grasp(a.closure)[0]
    camera = None
    if a.video:
        from isaaclab.sensors import Camera, CameraCfg

        camera = Camera(
            CameraCfg(
                prim_path="/World/Camera",
                height=960,
                width=960,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24,
                    focus_distance=0.4,
                    horizontal_aperture=20,
                    clipping_range=(0.01, 10),
                ),
            )
        )
    sim.reset()
    cube.reset()
    for h in hands:
        h.initialize(sim.device)
    contact_views = [
        h.robot._physics_sim_view.create_rigid_contact_view(
            h.path + "/" + h.hand.prefix + "*",
            filter_patterns=[
                "/World/Cube/" + n for n in (names if a.locked_cube else names[1:])
            ],
        )
        for h in hands
    ]
    print(
        "CONTACT VIEWS",
        [(v.sensor_count, v.filter_count) for v in contact_views],
        flush=True,
    )
    initial = cube.data.object_state_w.clone()
    initial[:, :, :3] = torch.tensor([0, 0, 0.5], device=sim.device)
    initial[:, :, 3:7] = torch.tensor(
        Rotation.from_matrix(reference_R).as_quat()[[3, 0, 1, 2]],
        device=sim.device,
        dtype=torch.float,
    )
    initial[:, :, 7:] = 0
    if not a.locked_cube:
        initial[0, 1:, 3:7] = torch.tensor(
            Rotation.from_matrix(reference_R @ state_after("U'")).as_quat()[
                :, [3, 0, 1, 2]
            ],
            device=sim.device,
            dtype=torch.float,
        )
    cube.write_object_state_to_sim(initial)
    allnames = names + left.robot.body_names + right.robot.body_names
    export_geometry(stage, allnames, a.output)
    if camera:
        camera.set_world_poses_from_view(
            torch.tensor([[0.25, -0.32, 0.76]], device=sim.device),
            torch.tensor([[0, 0, 0.51]], device=sim.device),
        )
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            str(a.output / "video.mp4"), fps=50, codec="libx264", quality=8
        )
    records = []
    samples = []
    expected_state = state_after("U'" if a.hold_only else "")
    device = sim.device
    turn_start = a.fixture_warmup + 1.5
    for step in range(round(a.duration / 0.001)):
        if a.fixture_warmup > 0 and step == round(a.fixture_warmup / 0.001):
            UsdPhysics.Joint(
                stage.GetPrimAtPath("/World/Cube/fixture")
            ).GetJointEnabledAttr().Set(False)
            print("FIXTURE RELEASED", flush=True)
        t = step * 0.001
        u = 0.0 if a.hold_only else np.clip((t - turn_start) / 4, 0, 1)
        angle = -np.pi / 2 * u * u * (3 - 2 * u)
        core_pose = cube.data.object_state_w[0, 0, :7].cpu().numpy()
        core_rotation = Rotation.from_quat(core_pose[[4, 5, 6, 3]]).as_matrix()
        closure_fraction = float(np.clip(t / max(0.5, a.fixture_warmup * 0.75), 0, 1))
        closure_fraction = closure_fraction**2 * (3 - 2 * closure_fraction)

        def close(hand, goal):
            return {
                n: hand.q.get(n, 0.0) + closure_fraction * (v - hand.q.get(n, 0.0))
                for n, v in goal.items()
            }

        left.command(joints=close(left, left_goal))
        right_frame = (
            core_rotation @ Rotation.from_rotvec([0, 0, angle]).as_matrix() @ Sr
        )
        virtual_center = core_pose[:3] + 0.01905 * (
            core_rotation[:, 2] - right_frame[:, 2]
        )
        if a.hold_only:
            right_frame, virtual_center = reference_R @ Sr, right_origin
        right.command(
            right_frame, joints=close(right, right_goal), cube_pos=virtual_center
        )
        states = cube.data.object_state_w
        rots = matrix_from_quat(states[0, :, 3:7])
        local = (
            torch.zeros((1, len(names), 3), device=device)
            if a.locked_cube
            else local_torques(rots, states[0, :, 10:13])[None]
        )
        cube.set_external_force_and_torque(torch.zeros_like(local), local)
        cube.write_data_to_sim()
        sim.step(render=a.video and step % 20 == 0)
        cube.update(0.001)
        for h in hands:
            h.robot.update(0.001)
        if step % 20 == 0:
            s = cube.data.object_state_w[0].cpu().numpy()
            r = Rotation.from_quat(s[:, [4, 5, 6, 3]]).as_matrix()
            rel = r[0].T @ r[1:]
            decoded = (
                decode(rel)
                if not a.locked_cube
                else dict(aligned=None, solved=None, facelets=None)
            )
            errors = (
                Rotation.from_matrix(
                    rel @ expected_state.transpose(0, 2, 1)
                ).magnitude()
                * 180
                / np.pi
                if not a.locked_cube
                else np.array([0.0])
            )
            samples.append(
                dict(
                    t=round(t, 3),
                    net_hand_contact_n={
                        side: float(
                            v.get_net_contact_forces(dt=0.001).norm(dim=-1).sum()
                        )
                        for side, v in zip(("left", "right"), contact_views)
                    },
                    contact_loads_n={
                        side: v.get_contact_force_matrix(dt=0.001)
                        .norm(dim=-1)
                        .sum(0)
                        .cpu()
                        .tolist()
                        for side, v in zip(("left", "right"), contact_views)
                    },
                    core_euler_deg=Rotation.from_matrix(r[0])
                    .as_euler("xyz", degrees=True)
                    .tolist(),
                    core_rotation_error_deg=float(
                        np.rad2deg(
                            Rotation.from_matrix(r[0] @ reference_R.T).magnitude()
                        )
                    ),
                    phase=(
                        "Prepare grasp"
                        if t < a.fixture_warmup
                        else (
                            "Free grasp"
                            if a.hold_only or t < turn_start
                            else "Turn" if t < turn_start + 4 else "Verify"
                        )
                    ),
                    move="Hold" if a.hold_only else "U",
                    **(
                        measure_cube(s)
                        if not a.locked_cube
                        else dict(
                            decoded=decoded,
                            anchors_connected=True,
                            max_anchor_error_m=0.0,
                        )
                    ),
                    expected_facelets=(
                        None if a.locked_cube else decode(expected_state)["facelets"]
                    ),
                    max_target_error_deg=float(errors.max()),
                    cube=s[0, :3].tolist(),
                    core_position_error_m=float(np.linalg.norm(s[0, :3] - [0, 0, 0.5])),
                    joint_torque_max_nm=max(
                        float(h.robot.data.applied_torque.abs().max()) for h in hands
                    ),
                )
            )
            records.append(
                np.concatenate(
                    [s] + [h.robot.data.body_state_w[0].cpu().numpy() for h in hands]
                )
            )
            if camera:
                camera.update(0.02)
                writer.append_data(camera.data.output["rgb"][0, :, :, :3].cpu().numpy())
            if samples[-1]["core_position_error_m"] > 0.15:
                print("STOP dropped cube", flush=True)
                break
        if step % 1000 == 0:
            print("PROGRESS", samples[-1], flush=True)
    if camera:
        writer.close()
    np.asarray(records, dtype="<f4")[:, :, :7].tofile(a.output / "poses.bin")
    np.savez_compressed(
        a.output / "poses.npz", states=np.array(records), names=np.array(allnames)
    )
    (a.output / "telemetry.json").write_text(json.dumps(samples))
    hold_result = (
        evaluate_hold(samples, a.fixture_warmup, a.duration) if a.hold_only else None
    )
    result = dict(
        task=(
            "Unsupported stationary grasp"
            if a.hold_only
            else "Bimanual U inverse solve"
        ),
        engine="Isaac Lab / PhysX",
        fixture=False,
        locked_cube=a.locked_cube,
        grasp_validation=hold_result,
        initial_fixture_seconds=a.fixture_warmup,
        cube_actuators=0,
        contact_model=right.contact_model,
        cube_state_writes_during_rollout=0,
        frames=len(records),
        bodies=len(allnames),
        fps=50,
        duration=(len(records) - 1) * 0.02,
        physics_dt_s=0.001,
        passed=(
            hold_result["passed"]
            if a.hold_only
            else bool(
                stable_match(samples, decode(state_after(""))["facelets"])
                and samples[-1]["core_position_error_m"] < 0.02
            )
        ),
        configuration=vars(a),
        final=samples[-1],
    )
    (a.output / "result.json").write_text(json.dumps(result, indent=2, default=str))
    stage.Export(str(a.output / "scene.usda"))
    print("RESULT", json.dumps(result, default=str), flush=True)


main()
import threading, os

threading.Timer(5, lambda: os._exit(0)).start()
app.close()
