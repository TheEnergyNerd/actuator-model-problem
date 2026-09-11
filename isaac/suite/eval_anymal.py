"""Evaluate a compatible RSL-RL rough-terrain policy with measured state export.

Native terrain and command sampling are retained. This measures a baseline,
not completion of a bespoke obstacle course. Episode resets are retained and
explicitly recorded. No success threshold is inferred from reward.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import threading

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--seconds", type=float, default=20)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=201)
parser.add_argument(
    "--design",
    choices=(
        "nominal",
        "kv_high",
        "kv_low",
        "gear_low",
        "gear_high",
        "torque_low",
        "mass_added_double",
    ),
    default="nominal",
)
parser.add_argument(
    "--course",
    action="store_true",
    help="Evaluate the authored stairs/platform/block route with one robot",
)
parser.add_argument(
    "--record",
    action="store_true",
    help="Export env 0 geometry and scene for synchronized video/replay",
)
parser.add_argument(
    "--atlas",
    action="store_true",
    help="Use Atlas FOC legs with the same stock policy and PLAY task",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.seconds <= 0 or args.num_envs < 1 or not args.checkpoint.is_file():
    parser.error(
        "positive duration/environment count and an existing checkpoint are required"
    )
if args.course and args.num_envs != 1:
    parser.error("The authored course requires --num-envs 1")
if args.design != "nominal" and not args.atlas:
    parser.error("Design interventions require --atlas")
args.output.mkdir(parents=True, exist_ok=False)
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry

env = None
code = 1
try:
    task = "Isaac-Velocity-Rough-Anymal-C-Play-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    if args.course:
        from course import configure_course, FINISH_X

        configure_course(cfg, args.output / "course.usda")
    if args.atlas:
        from atlas_actuators.tasks.anymal_rough_atlas import AnymalCRoughAtlasFOCEnvCfg

        # Change only the actuator model; retain identical PLAY events/terrain/commands.
        cfg.scene.robot.actuators = AnymalCRoughAtlasFOCEnvCfg().scene.robot.actuators
    agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent.device = args.device
    (args.output / "config.txt").write_text(str(cfg.to_dict()))
    base = gym.make(task, cfg=cfg)
    env = RslRlVecEnvWrapper(base, clip_actions=agent.clip_actions)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device=args.device)
    obs, _ = env.get_observations()
    raw = base.unwrapped
    robot = raw.scene["robot"]
    design = None
    if args.atlas:
        from designs import apply_design

        design = apply_design(robot, args.design)
    dt = raw.step_dt
    frames, rows = [], []
    finish_hold = 0.0
    course_passed = False
    course_violation = None
    # Keep the pre-step command: a reset may resample it inside env.step().
    with torch.inference_mode():
        for step in range(round(args.seconds / dt)):
            if args.course and float(robot.data.root_pos_w[0, 0]) >= FINISH_X:
                raw.command_manager.get_command("base_velocity")[:] = 0
            command = raw.command_manager.get_command("base_velocity").clone()
            obs, reward, done, info = env.step(policy(obs))
            poses = torch.cat((robot.data.body_pos_w, robot.data.body_quat_w), dim=-1)
            if not torch.isfinite(poses).all() or not torch.isfinite(obs).all():
                raise ValueError(f"Non-finite state at step {step}")
            frames.append(poses.cpu().numpy())
            rows.append(
                {
                    "t": (step + 1) * dt,
                    "command": command.cpu().tolist(),
                    "velocity": robot.data.root_lin_vel_b.cpu().tolist(),
                    "yaw_rate": robot.data.root_ang_vel_b[:, 2].cpu().tolist(),
                    "terminated": raw.reset_terminated.cpu().tolist(),
                    "time_out": raw.reset_time_outs.cpu().tolist(),
                    "temperature_c": (
                        max(float(a._winding_T.max()) for a in robot.actuators.values())
                        if args.atlas
                        else None
                    ),
                    "peak_joint_torque_nm": float(
                        robot.data.applied_torque.abs().max()
                    ),
                    "peak_current_a": (
                        max(float(a._iq.abs().max()) for a in robot.actuators.values())
                        if args.atlas
                        else None
                    ),
                }
            )
            if args.course:
                failed = bool(raw.reset_terminated.any() or raw.reset_time_outs.any())
                route_x = float(robot.data.root_pos_w[0, 0])
                lateral_limit = 1.3 if 2 <= route_x <= 10.7 else 2.3
                if abs(float(robot.data.root_pos_w[0, 1])) > lateral_limit:
                    course_violation = "Left the obstacle corridor or approach/finish floor"
                    failed = True
                stopped = (
                    float(robot.data.root_pos_w[0, 0]) >= FINISH_X
                    and float(robot.data.root_lin_vel_w[0, :2].norm()) < 0.1
                )
                finish_hold = finish_hold + dt if stopped else 0.0
                course_passed = finish_hold >= 1.0 and not failed
                if failed or course_passed:
                    break
    # Post-reset velocities do not describe the pre-reset transition; exclude them.
    valid = np.array(
        [[not (a or b) for a, b in zip(r["terminated"], r["time_out"])] for r in rows]
    )
    command = np.array([r["command"] for r in rows])
    velocity = np.array([r["velocity"] for r in rows])
    error = np.linalg.norm(command[:, :, :2] - velocity[:, :, :2], axis=-1)
    summary = {
        "kind": "course_evaluation" if args.course else "stock_policy_baseline",
        "actuator_model": "atlas_foc" if args.atlas else "native_anydrive",
        "design": design,
        "task": task,
        "task_success": course_passed if args.course else None,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "duration_s": len(rows) * dt,
        "frames": len(rows),
        "bodies": robot.body_names,
        "physics_dt": cfg.sim.dt,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "termination_count": np.sum([r["terminated"] for r in rows], axis=0).tolist(),
        "timeout_count": np.sum([r["time_out"] for r in rows], axis=0).tolist(),
        "mean_xy_tracking_error_m_s": [
            float(error[:, i][valid[:, i]].mean()) if valid[:, i].any() else None
            for i in range(args.num_envs)
        ],
        "reset_policy": "Native resets retained; terminated and time_out flags mark discontinuities.",
        "terrain": (
            "Authored stairs, platform and block course"
            if args.course
            else "Native 5x5 PLAY terrain; not an authored obstacle course."
        ),
        "course_finish_x_m": FINISH_X if args.course else None,
        "course_finish_hold_s": finish_hold if args.course else None,
        "course_violation": course_violation,
    }
    np.savez_compressed(
        args.output / "poses.npz", poses=np.asarray(frames, dtype=np.float32)
    )
    if args.record:
        from recording import export_recording

        export_recording(args.output, raw, np.asarray(frames), rows, summary)
    (args.output / "telemetry.json").write_text(json.dumps(rows, allow_nan=False))
    (args.output / "result.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print("BASELINE_RESULT " + json.dumps(summary), flush=True)
    code = 0
finally:
    timer = threading.Timer(10, lambda: os._exit(code))
    timer.daemon = True
    timer.start()
    if env is not None:
        env.close()
    app.close()
