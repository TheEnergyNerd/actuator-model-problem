"""Contact-driven Wuji face-turn diagnostic in Isaac Lab (fixture, not a full solve)."""

import argparse, json, math
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--seconds", type=float, default=8)
p.add_argument("--closure", type=float, default=0.004)
p.add_argument("--detent", type=float, default=0.02)
p.add_argument("--guide", type=float, default=0.2)
p.add_argument("--joint-damping", type=float, default=0.0001)
p.add_argument("--feedback", action="store_true")
p.add_argument("--contact-profile", choices=("stock", "rigid", "pad"), default="stock")
p.add_argument("--scramble", default="")
p.add_argument("--moves", default="")
p.add_argument("--video", action="store_true")
p.add_argument("--overtravel-deg", type=float, default=0.0)
p.add_argument("--max-retries", type=int, default=3)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if not 0 <= a.closure <= 0.012 or not 0 <= a.overtravel_deg <= 90:
    p.error("closure must be 0–0.012 m and overtravel must be 0–90 degrees")
if a.max_retries < 0 or a.seconds <= 0:
    p.error("max-retries must be nonnegative and seconds must be positive")
a.enable_cameras = a.video
app = AppLauncher(a).app
import numpy as np, torch
from scipy.spatial.transform import Rotation
from pxr import UsdPhysics, PhysxSchema, UsdShade
import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.math import matrix_from_quat, quat_rotate_inverse
from cube_scene import make_cube
from cube_state import COORDS, decode, state_after, apply_move, NORMALS, FACES
from sequence_control import Sequence
from passive import local_torques
from validation import measure_cube, stable_match
from contact_model import configure_hand_contacts
from wuji_kinematics import Hand


def main():
    a.output.mkdir(parents=True, exist_ok=False)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=0.001,
            device=a.device,
            gravity=(0, 0, -9.81),
            render_interval=20,
            physx=sim_utils.PhysxCfg(
                solver_type=1,
                min_position_iteration_count=32,
                min_velocity_iteration_count=8,
            ),
        )
    )
    stage = sim.stage
    sim_utils.DomeLightCfg(intensity=1800).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=1800)
    )
    names = make_cube(stage, scramble=a.scramble)
    cube = RigidObjectCollection(
        RigidObjectCollectionCfg(
            rigid_objects={
                n: RigidObjectCfg(prim_path="/World/Cube/" + n, spawn=None)
                for n in names
            }
        )
    )
    hand = Hand("right")
    open_q, _ = hand.grasp(-0.005)
    closed_q, ik_error = hand.grasp(a.closure)
    sequence = Sequence(hand, a.moves, a.closure) if a.moves else None
    if sequence:
        sequence.angles = [
            angle + np.sign(angle) * np.deg2rad(a.overtravel_deg)
            for angle in sequence.angles
        ]
        pos, w, open_q, _, _ = sequence.command(0)
        a.seconds = len(sequence.moves) * sequence.period
    else:
        pos, w = hand.wrist_pose([0, 0, 0.5])
    quat = Rotation.from_matrix(w).as_quat()[[3, 0, 1, 2]]
    usd = str(
        Path(__file__).parent
        / "assets/wuji/hand2/hand2_beta1/body/usd/right/wujihand2.usd"
    )
    cfg = ArticulationCfg(
        prim_path="/World/Hand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=0.2
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=8,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(pos), rot=tuple(quat), joint_pos=open_q
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=3.0,
                damping=0.08,
                effort_limit_sim=1.5,
            )
        },
    )
    robot = Articulation(cfg)
    # Convert the upstream fixed mounting to a floating wrist controlled by a bounded external wrench.
    for prim in list(stage.Traverse()):
        if str(prim.GetPath()).startswith("/World/Hand") and prim.IsA(
            UsdPhysics.FixedJoint
        ):
            j = UsdPhysics.Joint(prim)
            if not j.GetBody0Rel().GetTargets():
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                prim.SetActive(False)
    wrist = stage.GetPrimAtPath("/World/Hand/r_wrist")
    UsdPhysics.ArticulationRootAPI.Apply(wrist)
    contact_model = configure_hand_contacts(
        stage, "/World/Hand", profile=a.contact_profile
    )
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
    robot.reset()
    cube.reset()
    robot.write_root_pose_to_sim(
        torch.tensor([list(pos) + list(quat)], device=sim.device, dtype=torch.float)
    )
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos, robot.data.default_joint_vel
    )
    initial = cube.data.object_state_w.clone()
    initial[:, :, :3] = torch.tensor([0, 0, 0.5], device=sim.device)
    initial[:, :, 3:7] = torch.tensor([1, 0, 0, 0], device=sim.device)
    initial[:, :, 7:] = 0
    initial[0, 1:, 3:7] = torch.tensor(
        Rotation.from_matrix(state_after(a.scramble)).as_quat()[:, [3, 0, 1, 2]],
        device=sim.device,
        dtype=torch.float,
    )
    cube.write_object_state_to_sim(initial)
    if camera:
        camera.set_world_poses_from_view(
            torch.tensor([[0.23, -0.28, 0.72]], device=sim.device),
            torch.tensor([[0, 0, 0.48]], device=sim.device),
        )
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            str(a.output / "video.mp4"), fps=50, codec="libx264", quality=8
        )
    ids = [names.index(f"cubie_{i:02d}") for i in range(26)]
    wrist_id = robot.body_names.index("r_wrist")
    target = robot.data.default_joint_pos.clone()
    records = []
    telemetry = []
    device = sim.device
    core = names.index("core")
    from export_replay import export_geometry

    export_geometry(stage, names + robot.body_names, a.output)
    print("READY", robot.joint_names, robot.body_names, flush=True)
    bypass_feedback = False
    attempts = {}
    completed_moves = 0
    control_t = 0.0
    blocked = False
    blocked_time = 0.0
    grip_level = 0
    last_move = 0
    grips = [
        hand.grasp(min(0.012, a.closure + i * 0.001))[0]
        for i in range(max(1, round((0.012 - a.closure) / 0.001) + 1))
    ]
    max_seconds = (
        (
            len(sequence.moves)
            * (a.max_retries + 1)
            * (sequence.period + (10 if a.feedback else 0))
        )
        if sequence
        else a.seconds
    )
    for step in range(round(max_seconds / 0.001)):
        t = step * 0.001
        close = np.clip(t / 0.8, 0, 1)
        for name, value in closed_q.items():
            target[0, robot.joint_names.index(name)] = open_q[name] + close * (
                value - open_q[name]
            )
        robot.set_joint_position_target(target)
        # One U turn, with smooth wrist trajectory; this is a commanded hand motion, never a cube state write.
        u = np.clip((t - 1.5) / 4, 0, 1)
        angle = -math.pi / 2 * (3 * u * u - 2 * u * u * u)
        turn = Rotation.from_rotvec([0, 0, angle]).as_matrix()
        des_pos, des_w = hand.wrist_pose([0, 0, 0.5], turn)
        move_index = 0
        phase = "Turn" if t < 5.5 else "Verify"
        if sequence:
            des_pos, des_w, joints, move_index, phase = sequence.command(control_t)
            if move_index != last_move:
                grip_level = 0
                blocked_time = 0.0
                last_move = move_index
                bypass_feedback = False
            sequence.closed = grips[grip_level]
            for name, value in joints.items():
                target[0, robot.joint_names.index(name)] = value
            robot.set_joint_position_target(target)
            commanded = sequence.frames[move_index].T @ des_w @ hand.Q
            angle = math.atan2(commanded[1, 0], commanded[0, 0])
        body = robot.data.body_state_w[0, wrist_id]
        cur_w = matrix_from_quat(body[3:7]).cpu().numpy()
        er = Rotation.from_matrix(des_w @ cur_w.T).as_rotvec()
        force = (torch.tensor(des_pos, device=device) - body[:3]) * 600 - body[
            7:10
        ] * 14
        torque = torch.tensor(er, device=device) * 4 - body[10:13] * 0.14
        force = torch.clamp(force, -30, 30).float()
        torque = torch.clamp(torque, -2, 2).float()
        robot.set_external_force_and_torque(
            quat_rotate_inverse(body[3:7], force)[None, None, :],
            quat_rotate_inverse(body[3:7], torque)[None, None, :],
            body_ids=[wrist_id],
        )
        states = cube.data.object_state_w
        rotations = matrix_from_quat(states[0, :, 3:7])
        torques = local_torques(
            rotations, states[0, :, 10:13], a.detent, a.guide, a.joint_damping
        )[None]
        cube.set_external_force_and_torque(torch.zeros_like(torques), torques)
        robot.write_data_to_sim()
        cube.write_data_to_sim()
        sim.step(render=a.video and step % 20 == 0)
        robot.update(0.001)
        cube.update(0.001)
        if step % 20 == 0:
            s = cube.data.object_state_w[0].cpu().numpy()
            r = Rotation.from_quat(s[:, [4, 5, 6, 3]]).as_matrix()
            relative = r[core].T @ r[ids]
            expected = state_after(
                a.scramble
                + " "
                + (" ".join(sequence.moves[: move_index + 1]) if sequence else "U")
            )
            errors = (
                Rotation.from_matrix(relative @ expected.transpose(0, 2, 1)).magnitude()
                * 180
                / np.pi
            )
            blocked = False
            if sequence and a.feedback and not bypass_feedback and phase == "Turn":
                before = state_after(
                    a.scramble + " " + " ".join(sequence.moves[:move_index])
                )
                normal = NORMALS[FACES.index(sequence.moves[move_index][0])]
                mask = np.einsum("nij,nj->ni", before, COORDS) @ normal > 0.5
                actual = float(
                    np.median(
                        Rotation.from_matrix(
                            relative[mask] @ before[mask].transpose(0, 2, 1)
                        ).as_rotvec()
                        @ normal
                    )
                )
                blocked = np.sign(sequence.angles[move_index]) * (
                    angle - actual
                ) > np.deg2rad(50)
                if blocked:
                    blocked_time += 0.02
                    if blocked_time > 0.4 and grip_level < len(grips) - 1:
                        grip_level += 1
                        blocked_time = 0.0
                else:
                    blocked_time = 0.0
            telemetry.append(
                {
                    "t": round(t, 3),
                    "cube": s[0, :3].tolist(),
                    "phase": phase,
                    "move_index": move_index,
                    "attempt": attempts.get(move_index, 0) + 1,
                    "move": sequence.moves[move_index] if sequence else "U",
                    "expected_facelets": decode(expected)["facelets"],
                    "command_angle_deg": angle * 180 / np.pi,
                    "max_target_error_deg": float(errors.max()),
                    "controller_time_s": control_t,
                    "grip_closure_m": min(0.012, a.closure + grip_level * 0.001),
                    "progress_limited": bool(blocked),
                    "mean_target_error_deg": float(errors.mean()),
                    **measure_cube(s),
                    "wrist_position_error_m": float(
                        np.linalg.norm(des_pos - body[:3].cpu().numpy())
                    ),
                    "joint_torque_max_nm": float(robot.data.applied_torque.abs().max()),
                }
            )
            if (
                sequence
                and a.overtravel_deg
                and phase == "Turn"
                and stable_match(
                    telemetry, telemetry[-1]["expected_facelets"], hold_seconds=0.08
                )
            ):
                # Stop at the measured cube goal. Keep the current wrist pose continuous.
                sequence.angles[move_index] = angle
                control_t = move_index * sequence.period + 8.3
                blocked = False
                blocked_time = 0.0
                print(
                    "MEASURED TURN GOAL",
                    move_index,
                    float(np.rad2deg(angle)),
                    flush=True,
                )
            records.append(
                np.concatenate([s, robot.data.body_state_w[0].cpu().numpy()])
            )
            if camera:
                camera.update(0.02)
                writer.append_data(camera.data.output["rgb"][0, :, :, :3].cpu().numpy())
        if a.feedback and blocked_time > 3 and grip_level == len(grips) - 1:
            bypass_feedback = True
            blocked = False
        if sequence:
            if not blocked:
                control_t += 0.001
            boundary = (move_index + 1) * sequence.period
            if control_t >= boundary:
                matched = stable_match(telemetry, telemetry[-1]["expected_facelets"])
                if matched:
                    completed_moves = move_index + 1
                    print("MOVE PASSED AFTER RELEASE", completed_moves, flush=True)
                    if completed_moves == len(sequence.moves):
                        break
                elif attempts.get(move_index, 0) < a.max_retries:
                    previous = (
                        sequence.frames[move_index]
                        @ Rotation.from_rotvec(
                            [0, 0, sequence.angles[move_index]]
                        ).as_matrix()
                    )
                    sequence.entry_frames[move_index] = previous
                    sequence.frames[move_index] = (
                        sequence.frames[move_index]
                        @ Rotation.from_rotvec([0, 0, np.pi / 2]).as_matrix()
                    )
                    attempts[move_index] = attempts.get(move_index, 0) + 1
                    control_t = move_index * sequence.period
                    grip_level = 0
                    blocked_time = 0.0
                    blocked = False
                    bypass_feedback = False
                    print("REGRASP RETRY", move_index, attempts[move_index], flush=True)
                else:
                    print(
                        f"STOP: {a.max_retries + 1} grasp attempts failed measured validation",
                        move_index,
                        flush=True,
                    )
                    break
        if step % 1000 == 0:
            print("PROGRESS", step, telemetry[-1], flush=True)
    if camera:
        writer.close()
    np.asarray(records, dtype="<f4")[:, :, :7].tofile(a.output / "poses.bin")
    np.savez_compressed(
        a.output / "poses.npz",
        states=np.array(records),
        names=np.array(names + robot.body_names),
    )
    (a.output / "telemetry.json").write_text(json.dumps(telemetry))
    result = {
        "task": "fixture Rubik sequence" if sequence else "fixture U face turn",
        "engine": "Isaac Lab / PhysX",
        "controller": "URDF fingertip IK + bounded floating wrist PD",
        "cube_actuators": 0,
        "contact_model": contact_model,
        "cube_state_writes_during_rollout": 0,
        "fixture": True,
        "ik_error_m": ik_error,
        "frames": len(records),
        "bodies": len(names) + len(robot.body_names),
        "fps": 50,
        "duration": (len(records) - 1) * 0.02,
        "physics_dt_s": 0.001,
        "completed_moves": completed_moves,
        "attempts": attempts,
        "passed": stable_match(telemetry, telemetry[-1]["expected_facelets"])
        and (not sequence or completed_moves == len(sequence.moves)),
        "configuration": vars(a) | {"output": str(a.output)},
        "final": telemetry[-1],
    }
    (a.output / "result.json").write_text(json.dumps(result, indent=2, default=str))
    stage.Export(str(a.output / "scene.usda"))
    print("RESULT", json.dumps(result, default=str), flush=True)


main()
import threading, os

threading.Timer(5, lambda: os._exit(0)).start()
app.close()
