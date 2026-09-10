"""Paired actuator-design interventions in actual Isaac Lab environments.

One policy is held fixed. Designs are batched across environments; each design
receives the same set of initial physical properties. This measures policy
transfer sensitivity, not the optimum achievable after retraining each design.
"""

import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--robot", choices=["anymal", "g1", "allegro"], required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--task", default=None)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--replicas", type=int, default=32)
p.add_argument("--seconds", type=float, default=60)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--cases", default="all")
p.add_argument("--video", action="store_true")
p.add_argument("--replay", action="store_true", help="Export rigid-body poses, meshes and synchronized telemetry")
p.add_argument("--side-view", action="store_true")
p.add_argument("--physics-substeps", type=int, default=1)
p.add_argument(
    "--speed",
    type=float,
    default=None,
    help="Override locomotion schedule with a constant speed",
)
p.add_argument("--push-time", type=float, default=45.0)
p.add_argument(
    "--g1-explicit-gains",
    action="store_true",
    help="Use lower damping appropriate to explicit integration",
)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.replicas < 1 or a.seconds <= 0 or a.physics_substeps < 1:
    p.error("Positive replicas and seconds required")
a.headless = True
a.enable_cameras = a.video
app = AppLauncher(a).app

from dataclasses import asdict, replace
from functools import wraps
import hashlib, json, math
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks
import atlas_actuators
import atlas_actuators.tasks
from atlas_actuators.design_study import (
    designs,
    parameter_batch,
    describe,
    add_carrier_mass,
)
from atlas_actuators.foc_core import _Rs_at, _torque_motor
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

TASKS = {
    "anymal": ("Isaac-Velocity-Flat-Anymal-C-AtlasFOC-v0", 0.25, 1e-4, 0.04, 100.0),
    "g1": ("Isaac-Velocity-Flat-G1-AtlasFOC-v0", 0.25, 1e-4, 0.04, 80.0),
    "allegro": ("Isaac-Repose-Cube-Allegro-GearedReal20-v1", 0.01, 2e-6, 0.012, 0.08),
}


def main():
    a.output.mkdir(parents=True, exist_ok=False)
    task, mass_unit, rotor_unit, radius, push_N = TASKS[a.robot]
    task = a.task or task
    cases = designs()
    if a.replay:
        cases += [replace(cases[0], name="torque_very_low", axis="peak current stress", peak_current_factor=0.15),
                  replace(cases[0], name="kv_very_high", axis="rewind Kv stress", kv_factor=4.0)]
    if a.cases != "all":
        requested = a.cases.split(",")
        lookup = {d.name: d for d in cases}
        cases = [lookup[name] for name in requested]
    count = len(cases) * a.replicas
    if a.replay and (a.replicas != 1 or a.robot == "allegro"):
        raise ValueError("Locomotion replay requires one replica and a locomotion robot")
    if a.video and (a.replicas != 1 or len(cases) > (5 if a.replay else 3)):
        raise ValueError("Video requires one replica and at most three cases (five for replay)")
    cfg = parse_env_cfg(task, num_envs=count)
    cfg.seed = a.seed
    cfg.sim.dt /= a.physics_substeps
    cfg.decimation *= a.physics_substeps
    cfg.sim.render_interval = cfg.decimation
    for actuator_cfg in cfg.scene.robot.actuators.values():
        actuator_cfg.sim_dt = cfg.sim.dt
    if hasattr(cfg.scene, "contact_forces"):
        cfg.scene.contact_forces.update_period = cfg.sim.dt
    cfg.observations.policy.enable_corruption = False
    if a.g1_explicit_gains:
        if a.robot != "g1":
            raise ValueError("G1 explicit gains only apply to G1")
        from atlas_actuators.tasks.g1_atlas import explicit_arm_damping

        cfg.scene.robot.actuators["arms"].damping = explicit_arm_damping()

    # Isolate commanded tests from unrelated interval shoves and reset impulses.
    for name in ("push_robot", "base_external_force_torque"):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    if a.robot != "allegro":
        cmd = cfg.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (1.0, 1.0)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.resampling_time_range = (1e9, 1e9)
        if hasattr(cfg.events, "reset_base"):
            cfg.events.reset_base.params["pose_range"] = {}
            cfg.events.reset_base.params["velocity_range"] = {}
        if hasattr(cfg.events, "reset_robot_joints"):
            cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
            cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    else:
        cfg.events.reset_object.params["pose_range"] = {}
    if a.video:
        cfg.viewer.resolution = (800, 608)
        cfg.scene.env_spacing = 20.0
        if a.robot != "allegro":
            cfg.commands.base_velocity.debug_vis = False
        cfg.viewer.eye = (2.8, 2.8, 1.8) if a.robot != "allegro" else (0.65, 0.45, 0.9)
        cfg.viewer.lookat = (0, 0, 0.5) if a.robot != "allegro" else (0, -0.14, 0.46)
    agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent.seed = a.seed
    accum = {}
    windows = []
    current_step = [0]
    active = [False]
    window_accum = {}
    window_steps = [0]
    latest = {}
    video_rows = []
    writer = None
    reward_cfg = (
        cfg.rewards.success_bonus
        if a.robot == "allegro"
        else cfg.rewards.track_lin_vel_xy_exp
    )
    original = reward_cfg.func

    @wraps(original)
    def observe(env, **kwargs):
        value = original(env, **kwargs)
        if not active[0]:
            return value
        robot = env.scene["robot"]
        dt = env.step_dt
        sample = {
            "failures": env.termination_manager.terminated.float(),
            "timeouts": env.termination_manager.time_outs.float(),
        }
        if a.robot == "allegro":
            sample["goals"] = (value.bool() & ~env.reset_buf.bool()).float()
            command = env.command_manager.get_term("object_pose")
            from isaaclab.utils.math import quat_error_magnitude

            sample["orientation_error_rad"] = quat_error_magnitude(
                env.scene["object"].data.root_quat_w, command.quat_command_w
            )
        else:
            cmd = env.command_manager.get_term("base_velocity").vel_command_b
            vel = robot.data.root_lin_vel_b
            error = (vel[:, :2] - cmd[:, :2]).square().sum(-1)
            upright = robot.data.projected_gravity_b[:, 2] < -math.cos(math.pi / 4)
            sample["velocity_error_squared"] = error
            sample["tracking_score"] = torch.exp(-error / 0.25) * upright
            sample["forward_speed_m_s"] = vel[:, 0]
            sample["distance_m"] = vel[:, 0].clamp(min=0) * dt * upright
            sample["speed_on_target"] = (
                (vel[:, 0] - cmd[:, 0]).abs() < 0.2
            ).float() * upright
        if a.robot == "g1":
            from atlas_actuators.gait import gait_metrics

            sample.update(gait_metrics(env))
        copper = torch.zeros(count, device=env.device)
        draw = copper.clone()
        sat = copper.clone()
        temp = torch.zeros_like(copper)
        joints = 0
        for actuator in robot.actuators.values():
            par = actuator._p
            idx = actuator.joint_indices
            omega = robot.data.joint_vel[:, idx] * par.gear_ratio
            loss = (
                1.5
                * _Rs_at(par, actuator._winding_T)
                * (actuator._id.square() + actuator._iq.square())
            )
            shaft = (
                _torque_motor(par, actuator._id, actuator._iq, actuator._winding_T)
                * omega
            )
            copper += loss.sum(-1)
            draw += (loss + shaft).clamp(min=0).sum(-1)
            sat += (
                (actuator.computed_effort - actuator.applied_effort).abs()
                > 0.05 * actuator.computed_effort.abs().clamp(min=0.1)
            ).sum(-1)
            temp = torch.maximum(temp, actuator._winding_T.max(-1).values)
            joints += actuator.num_joints
        sample.update(
            copper_power_W=copper,
            estimated_positive_motor_input_W=draw,
            energy_J=draw * dt,
            torque_shortfall_fraction=sat / joints,
            temperature_C=temp,
        )
        latest.update(sample)
        for key, val in sample.items():
            if key not in accum:
                accum[key] = torch.zeros_like(val)
            accum[key].add_(val)
            if key not in window_accum:
                window_accum[key] = torch.zeros_like(val)
            window_accum[key].add_(val)
        window_steps[0] += 1
        # Lifetime counts and each replicate's scores are retained separately.
        if "ever_failed" not in accum:
            accum["ever_failed"] = torch.zeros(
                count, dtype=torch.bool, device=env.device
            )
        accum["ever_failed"] |= sample["failures"].bool()
        if "peak_temperature_C" not in accum:
            accum["peak_temperature_C"] = temp.clone()
        accum["peak_temperature_C"] = torch.maximum(accum["peak_temperature_C"], temp)
        if (current_step[0] + 1) % max(1, round(5 / dt)) == 0:
            rows = {
                key: (
                    val / window_steps[0]
                    if key
                    not in (
                        "failures",
                        "timeouts",
                        "goals",
                        "energy_J",
                        "distance_m",
                        "left_touchdowns",
                        "right_touchdowns",
                        "left_sustained_swing_landings",
                        "right_sustained_swing_landings",
                    )
                    else val
                )
                .reshape(len(cases), a.replicas)
                .mean(-1)
                .cpu()
                .tolist()
                for key, val in window_accum.items()
            }
            for val in window_accum.values():
                val.zero_()
            window_steps[0] = 0
            rows["time_s"] = (current_step[0] + 1) * dt
            windows.append(rows)
            print(f'DESIGN_PROGRESS {a.robot} {rows["time_s"]:.1f}s', flush=True)
        return value

    reward_cfg.func = observe
    raw = gym.make(task, cfg=cfg, render_mode="rgb_array" if a.video else None)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    try:
        base = env.unwrapped
        robot = base.scene["robot"]
        device = base.device

        def paired(tensor):
            return (
                tensor[: a.replicas]
                .repeat((len(cases),) + (1,) * (tensor.ndim - 1))
                .clone()
            )

        cpu_ids = torch.arange(count, dtype=torch.int32)
        # Copy sampled initial physical properties across design blocks.
        for asset in [robot] + ([base.scene["object"]] if a.robot == "allegro" else []):
            view = asset.root_physx_view
            view.set_masses(paired(view.get_masses()), cpu_ids)
            view.set_inertias(paired(view.get_inertias()), cpu_ids)
            view.set_coms(paired(view.get_coms()), cpu_ids)
            view.set_material_properties(
                paired(view.get_material_properties()), cpu_ids
            )
        specs = {}
        joint_gears = torch.ones(count, robot.num_joints, device=device)
        for name, actuator in robot.actuators.items():
            specs[name] = [
                describe(actuator._p, c, mass_unit, rotor_unit) for c in cases
            ]
            actuator.stiffness[:] = paired(actuator.stiffness)
            actuator.damping[:] = paired(actuator.damping)
            actuator._p = parameter_batch(actuator._p, cases, a.replicas, device)
            joint_gears[:, actuator.joint_indices] = actuator._p.gear_ratio
        # Resolve stator carriers from actual USD joint parent relationships.
        import omni.usd
        from pxr import Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        carriers = {}
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdPhysics.Joint) and prim.GetName() in robot.joint_names:
                parents = UsdPhysics.Joint(prim).GetBody0Rel().GetTargets()
                if parents:
                    name = parents[0].name
                    if name in robot.body_names:
                        carriers[prim.GetName()] = robot.body_names.index(name)
        if set(carriers) != set(robot.joint_names):
            raise RuntimeError(
                f"Unresolved actuator carriers: {set(robot.joint_names)-set(carriers)}"
            )
        mass_delta = torch.tensor(
            [c.added_mass_factor * mass_unit for c in cases]
        ).repeat_interleave(a.replicas)
        original_masses = robot.root_physx_view.get_masses().clone()
        masses, inertias = add_carrier_mass(
            original_masses,
            robot.root_physx_view.get_inertias(),
            [carriers[j] for j in robot.joint_names],
            mass_delta,
            radius,
        )
        robot.root_physx_view.set_masses(masses, cpu_ids)
        robot.root_physx_view.set_inertias(inertias, cpu_ids)
        measured_delta = robot.root_physx_view.get_masses().sum(
            -1
        ) - original_masses.sum(-1)
        torch.testing.assert_close(
            measured_delta, mass_delta * robot.num_joints, atol=1e-4, rtol=1e-4
        )
        rotor_delta = torch.tensor(
            [c.added_rotor_inertia_factor * rotor_unit for c in cases], device=device
        ).repeat_interleave(a.replicas)[:, None]
        armature = (
            paired(robot.data.joint_armature) + rotor_delta * joint_gears.square()
        )
        robot.write_joint_armature_to_sim(armature)
        torch.testing.assert_close(
            robot.root_physx_view.get_dof_armatures(), armature.cpu()
        )
        print(
            "DESIGN_PHYSICS_OK "
            + json.dumps(
                {
                    "robot": a.robot,
                    "joints": robot.num_joints,
                    "cases": len(cases),
                    "mass_delta_kg": measured_delta.reshape(len(cases), a.replicas)[
                        :, 0
                    ].tolist(),
                }
            ),
            flush=True,
        )
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(a.checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(device=device)
        forces = torch.zeros((count, robot.num_bodies, 3), device=device)
        torques = torch.zeros_like(forces)
        obj = base.scene["object"] if a.robot == "allegro" else robot
        if a.robot == "allegro":
            forces = torch.zeros((count, 1, 3), device=device)
            torques = torch.zeros_like(forces)
        if a.robot == "allegro":
            # Each replica gets the same goal sequence in every design, consumed
            # on success/reset. Keep Isaac's original two-angle distribution.
            import types
            from isaaclab.utils.math import quat_mul, quat_from_angle_axis, quat_unique

            command = base.command_manager.get_term("object_pose")
            generator = torch.Generator(device=device).manual_seed(a.seed + 1000)
            angles = (
                2
                * torch.rand((a.replicas, 4096, 2), generator=generator, device=device)
                - 1
            ) * torch.pi
            sequence_index = torch.zeros(count, dtype=torch.long, device=device)

            def resample_paired(term, env_ids):
                ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
                if len(ids) == 0:
                    return
                if (sequence_index[ids] >= 4096).any():
                    raise RuntimeError("Goal sequence exhausted")
                xy = angles[ids % a.replicas, sequence_index[ids]]
                quat = quat_mul(
                    quat_from_angle_axis(xy[:, 0], term._X_UNIT_VEC[ids]),
                    quat_from_angle_axis(xy[:, 1], term._Y_UNIT_VEC[ids]),
                )
                term.quat_command_w[ids] = (
                    quat_unique(quat) if term.cfg.make_quat_unique else quat
                )
                sequence_index[ids] += 1

            command._resample_command = types.MethodType(resample_paired, command)
        env.reset()
        if a.video:
            import imageio.v2 as imageio
            import numpy as np

            writer = imageio.get_writer(
                str(a.output / "comparison_raw.mp4"),
                fps=1 / (base.step_dt * 3),
                codec="libx264",
                quality=8,
            )
            base.render()
        recorder = None
        if a.replay:
            from locomotion_replay import LocomotionRecorder
            recorder = LocomotionRecorder(a.output, base, cases)
        active[0] = True
        obs, _ = env.get_observations()
        steps = round(a.seconds / base.step_dt)
        for step in range(steps):
            current_step[0] = step
            t = step * base.step_dt
            if a.robot != "allegro":
                target = (
                    a.speed
                    if a.speed is not None
                    else (0.5 if t < 20 else (2.0 if t < 40 else 1.0))
                )
                command = base.command_manager.get_term("base_velocity")
                command.vel_command_b[:] = 0
                command.vel_command_b[:, 0] = target
            forces.zero_()
            if a.push_time <= t < a.push_time + 0.2:
                forces[:, 0, 1] = push_N * torch.linspace(
                    0.8, 1.2, a.replicas, device=device
                ).repeat(len(cases))
            obj.set_external_force_and_torque(forces, torques)
            with torch.inference_mode():
                obs, _ = env.get_observations()
                obs, _, _, _ = env.step(policy(obs))
            if recorder is not None:
                recorder.capture(t, target, latest, accum["failures"], forces)
            if a.video and step % 3 == 0:
                frames = []
                for i, case in enumerate(cases):
                    if a.robot != "allegro":
                        pos = robot.data.root_pos_w[i].cpu().tolist()
                        # Follow the robot's heading so turning cannot turn a
                        # requested side view into a frontal view.
                        qw, qx, qy, qz = robot.data.root_quat_w[i].cpu().tolist()
                        yaw = math.atan2(
                            2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)
                        )
                        base.sim.set_camera_view(
                            (
                                [
                                    pos[0] - 2.6 * math.sin(yaw),
                                    pos[1] + 2.6 * math.cos(yaw),
                                    pos[2] + 0.35,
                                ]
                                if a.side_view
                                else [pos[0] + 1.8, pos[1] + 1.8, pos[2] + 0.85]
                            ),
                            [pos[0], pos[1], pos[2] - 0.15],
                        )
                    else:
                        origin = base.scene.env_origins[i].cpu().tolist()
                        base.sim.set_camera_view(
                            [origin[0] + 0.43, origin[1] + 0.26, origin[2] + 0.78],
                            [origin[0], origin[1] - 0.12, origin[2] + 0.51],
                        )
                    base.render()  # allow camera change to reach the renderer
                    frames.append(base.render().copy())
                writer.append_data(np.concatenate(frames, axis=1))
                video_rows.append(
                    {
                        "time_s": (step + 1) * base.step_dt,
                        "target_speed": target if a.robot != "allegro" else None,
                        "force_N": forces[:, 0, 1].cpu().tolist(),
                        **{key: val.cpu().tolist() for key, val in latest.items()},
                        "total_failures": accum["failures"].cpu().tolist(),
                        **(
                            {
                                "root_xy_m": (
                                    robot.data.root_pos_w[:, :2]
                                    - base.scene.env_origins[:, :2]
                                )
                                .cpu()
                                .tolist(),
                                "root_quat_wxyz": robot.data.root_quat_w.cpu().tolist(),
                            }
                            if a.robot != "allegro"
                            else {}
                        ),
                    }
                )
        if writer is not None:
            writer.close()
            writer = None
            (a.output / "video_telemetry.json").write_text(
                json.dumps(video_rows, allow_nan=False)
            )
        seconds = steps * base.step_dt
        records = []
        for i, case in enumerate(cases):
            sl = slice(i * a.replicas, (i + 1) * a.replicas)
            row = {
                "design": asdict(case),
                "groups": {k: v[i] for k, v in specs.items()},
                "total_robot_mass_kg": float(masses[sl].sum(-1).mean()),
                "added_total_mass_kg": float(measured_delta[sl].mean()),
                "fraction_without_failure": float(
                    (~accum["ever_failed"][sl]).float().mean()
                ),
                "failures": int(accum["failures"][sl].sum()),
                "peak_temperature_C": float(accum["peak_temperature_C"][sl].max()),
            }
            for key, val in accum.items():
                if key in ("ever_failed", "peak_temperature_C"):
                    continue
                scale = (
                    1
                    if key
                    in (
                        "goals",
                        "failures",
                        "timeouts",
                        "energy_J",
                        "distance_m",
                        "left_touchdowns",
                        "right_touchdowns",
                        "left_sustained_swing_landings",
                        "right_sustained_swing_landings",
                    )
                    else steps
                )
                row[key + "_per_replica"] = (val[sl] / scale).cpu().tolist()
                row[key + "_mean"] = float(val[sl].mean() / scale)
            if a.robot == "allegro":
                row["goals_per_hand_minute"] = row["goals_mean"] * 60 / seconds
            else:
                row["velocity_rmse_m_s"] = math.sqrt(row["velocity_error_squared_mean"])
                distance = row["distance_m_mean"]
                row["estimated_electrical_CoT"] = (
                    row["energy_J_mean"]
                    / (row["total_robot_mass_kg"] * 9.81 * distance)
                    if distance > 0.1
                    else None
                )
            records.append(row)
        result = {
            "robot": a.robot,
            "task": task,
            "policy_checkpoint": a.checkpoint,
            "checkpoint_sha256": hashlib.sha256(
                Path(a.checkpoint).read_bytes()
            ).hexdigest(),
            "physics_substeps": a.physics_substeps,
            "physics_dt_s": cfg.sim.dt,
            "speed_override": a.speed,
            "push_time_s": a.push_time,
            "g1_explicit_gains": a.g1_explicit_gains,
            "seed": a.seed,
            "replicas_per_design": a.replicas,
            "seconds": seconds,
            "records": records,
            "time_samples": windows,
            "carrier_mapping": {j: robot.body_names[b] for j, b in carriers.items()},
            "extension_path": atlas_actuators.__file__,
            "protocol": "Fixed policy; paired startup mass/inertia/COM/friction/gains; observation noise and unrelated pushes disabled. "
            "Normal episode resets remain. Locomotion commands 0.5 m/s for 20s, 2 m/s for 20s, then 1 m/s. "
            "Physical local-y force pulse at 45s for 0.2s. Allegro uses paired seeded orientation goal sequences consumed on success/reset. "
            "Added actuator housing mass is placed at parent-link COM with spherical-equivalent inertia. "
            "Added rotor inertia is reflected by gear ratio squared. Existing asset motor mass/inertia are not subtracted. "
            "Electrical power is a control-rate estimate from motor shaft power and copper loss, excluding auxiliary loads. "
            "Kv is phase-peak back-EMF rpm/V, not an unverified catalog convention. Hypothetical design sensitivities, not hardware validation.",
        }
        (a.output / "results.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        if recorder is not None:
            recorder.finish(result)
        print("DESIGN_STUDY_OK " + str(a.output), flush=True)
    finally:
        if writer is not None:
            writer.close()
        env.close()


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
    raise
finally:
    import threading, os
    timer = threading.Timer(15, lambda: os._exit(0))
    timer.daemon = True
    timer.start()
    app.close(wait_for_replicator=False)
