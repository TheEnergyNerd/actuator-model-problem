"""Detailed learned Allegro reorientation: real video + measured body replay."""

import argparse, json, hashlib, math, os, threading
from functools import wraps
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--label", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--seconds", type=float, default=24)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--precision-sequence", action="store_true")
p.add_argument("--terminal-hold", action="store_true")
p.add_argument("--physics-substeps", type=int, default=1)
p.add_argument("--finger-damping", type=float, default=None)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.output.exists():
    p.error("Output already exists")
if a.terminal_hold and not a.precision_sequence:
    p.error("Terminal hold requires --precision-sequence")
if a.physics_substeps < 1 or (a.finger_damping is not None and a.finger_damping < 0):
    p.error("Substeps must be positive and damping nonnegative")
a.headless = True
a.enable_cameras = True
app = AppLauncher(a).app
import torch, numpy as np, imageio.v2 as imageio, gymnasium as gym
import isaaclab_tasks, atlas_actuators.tasks
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.math import quat_error_magnitude
from locomotion_replay import LocomotionRecorder

TASK = (
    "Isaac-Repose-Cube-Allegro-Precision-v1"
    if a.precision_sequence
    else "Isaac-Repose-Cube-Allegro-GearedReal20-v1"
)
if a.precision_sequence:
    a.seconds = 60


def main():
    a.output.mkdir(parents=True)
    cfg = parse_env_cfg(TASK, num_envs=1)
    cfg.seed = a.seed
    cfg.observations.policy.enable_corruption = False
    cfg.sim.dt /= a.physics_substeps
    cfg.decimation *= a.physics_substeps
    cfg.sim.render_interval = cfg.decimation
    for actuator in cfg.scene.robot.actuators.values():
        actuator.sim_dt = cfg.sim.dt
        if a.finger_damping is not None:
            actuator.damping = a.finger_damping
    if a.precision_sequence:
        cfg.commands.object_pose.update_goal_on_success = False
    cfg.viewer.resolution = (1440, 1440)
    cfg.viewer.eye = (0.26, 0.08, 0.68)
    cfg.viewer.lookat = (0, -0.12, 0.50)
    cfg.scene.dome_light.spawn.color = (0.65, 0.68, 0.62)
    cfg.scene.dome_light.spawn.intensity = 1200
    cfg.scene.light.spawn.intensity = 2500
    cfg.commands.object_pose.debug_vis = False
    cfg.terminations.time_out = None
    cfg.terminations.max_consecutive_success = None
    agent = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent.seed = a.seed
    counters = {"goals": 0, "drops": 0}
    original = cfg.rewards.success_bonus.func
    observed = {}

    @wraps(original)
    def observe(env, **kw):
        reward = original(env, **kw)
        term = env.command_manager.get_term("object_pose")
        obj = env.scene["object"]
        dropped = bool(env.termination_manager.get_term("object_out_of_reach")[0])
        # Reward is evaluated before reset and goal resampling.
        success = bool(reward[0]) and not dropped
        if not a.precision_sequence:
            counters["goals"] += int(success)
        counters["drops"] += int(dropped)
        observed.update(
            target=term.quat_command_w[0].detach().cpu().tolist(),
            error=float(
                quat_error_magnitude(obj.data.root_quat_w, term.quat_command_w)[0]
            ),
            success=success,
            drop=dropped,
        )
        return reward

    cfg.rewards.success_bonus.func = observe
    raw = gym.make(TASK, cfg=cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    base = env.unwrapped
    try:
        from hand_studio import apply_materials

        apply_materials()
        robot = base.scene["robot"]
        obj = base.scene["object"]
        motor = robot.actuators["fingers"]
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(a.checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(device=base.device)
        env.reset()
        counters.update(goals=0, drops=0)
        recorder = LocomotionRecorder(a.output, base, [None])
        scene = json.loads((a.output / "scene.json").read_text())
        from pxr import UsdGeom, Usd
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        cube_prim = stage.GetPrimAtPath("/World/envs/env_0/object")
        bounds = (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            .ComputeUntransformedBound(cube_prim)
            .ComputeAlignedRange()
            .GetSize()
        )
        print("CUBE_DIMENSIONS", list(bounds), flush=True)
        scene.update(
            kind="hand",
            bodies=robot.body_names + ["cube"],
            cube_size=float(max(bounds)),
            follow_body=len(robot.body_names),
            joint_names=robot.joint_names,
        )
        for m in scene["meshes"]:
            name = robot.body_names[m["body"]]
            m["color"] = (
                [0.045, 0.06, 0.052]
                if name.endswith("_3")
                else [0.67, 0.71, 0.64] if "palm" in name else [0.80, 0.82, 0.76]
            )
        (a.output / "scene.json").write_text(json.dumps(scene, separators=(",", ":")))
        command = base.command_manager.get_term("object_pose")
        if a.precision_sequence:
            from isaaclab.utils.math import quat_mul, quat_from_angle_axis
            from atlas_actuators.tasks.allegro_precision import conditions

            initial = obj.data.root_quat_w.clone()
            axes = torch.eye(3, device=base.device)
            targets = []
            for axis, angle in [
                (0, 0),
                (1, torch.pi / 4),
                (1, torch.pi / 2),
                (0, torch.pi / 2),
                (2, torch.pi / 2),
                (0, 0),
            ]:
                targets.append(
                    quat_mul(
                        quat_from_angle_axis(
                            torch.tensor([angle], device=base.device),
                            axes[axis : axis + 1],
                        ),
                        initial,
                    )
                )
        from atlas_actuators.hold_controller import TerminalHold

        previous_action = torch.zeros((1, 16), device=base.device)
        terminal = TerminalHold(previous_action)
        dwell = 0.0
        stage_done = [False] * 6
        last_phase = -1
        forces = torch.zeros((1, 1, 3), device=base.device)
        torques = torch.zeros_like(forces)
        frames = []
        samples = []
        steps = round(a.seconds / base.step_dt)
        with imageio.get_writer(
            str(a.output / "video.mp4"),
            fps=1 / base.step_dt,
            codec="libx264",
            quality=7,
            macro_block_size=2,
        ) as video:
            for step in range(steps):
                t = step * base.step_dt
                phase = min(5, int(t / 10))
                stage_event = False
                if a.precision_sequence:
                    if phase != last_phase:
                        dwell = 0.0
                        command.hold_time.zero_()
                        last_phase = phase
                        terminal.reset()
                    command.quat_command_w[:] = targets[phase]
                    forces.zero_()
                    if 51 <= t < 51.2:
                        forces[0, 0, 0] = 0.08
                        dwell = 0.0
                        stage_done[5] = False
                    obj.set_external_force_and_torque(forces, torques)
                with torch.no_grad():
                    obs, _ = env.get_observations()
                    action = policy(obs)
                    if a.terminal_hold and a.precision_sequence:
                        action = terminal.command(
                            action,
                            previous_action,
                            quat_error_magnitude(obj.data.root_quat_w, targets[phase]),
                            obj.data.root_ang_vel_w.norm(dim=-1),
                        )
                    previous_action.copy_(action)
                    _, _, done, _ = env.step(action)
                    terminal.reset(done.bool())
                pose = (
                    torch.cat(
                        (
                            torch.cat(
                                (robot.data.body_pos_w[0], obj.data.root_pos_w), 0
                            ),
                            torch.cat(
                                (robot.data.body_quat_w[0], obj.data.root_quat_w), 0
                            ),
                        ),
                        1,
                    )
                    .cpu()
                    .numpy()
                )
                frames.append(pose)
                if a.precision_sequence:
                    *_, good = conditions(base)
                    dwell = (
                        dwell + base.step_dt
                        if bool(good[0]) and not observed["drop"]
                        else 0.0
                    )
                    if phase == 5 and t < 51.2:
                        dwell = 0.0
                    if dwell >= 0.5 - 1e-6 and not stage_done[phase]:
                        stage_done[phase] = True
                        stage_event = True
                    counters["goals"] = sum(stage_done)
                samples.append(
                    dict(
                        t=step * base.step_dt,
                        phase=(
                            [
                                "Initial hold",
                                "Rotate Y 45°",
                                "Rotate Y 90°",
                                "Rotate X 90°",
                                "Rotate Z 90°",
                                "Return and recover",
                            ][phase]
                            if a.precision_sequence
                            else "Reorient"
                        ),
                        cube=pose[-1, :3].tolist(),
                        tips=[],
                        grip=[],
                        temperature=float(motor._winding_T.max()),
                        torque=float(motor.applied_effort.abs().max()),
                        requested_torque=float(motor.computed_effort.abs().max()),
                        current=float(motor._iq.abs().max()),
                        saturation=float(
                            (
                                (motor.computed_effort - motor.applied_effort).abs()
                                > 0.05 * motor.computed_effort.abs().clamp(min=0.1)
                            )
                            .float()
                            .mean()
                        ),
                        failures=counters["drops"],
                        goals=counters["goals"],
                        goal_reached=(
                            stage_event if a.precision_sequence else observed["success"]
                        ),
                        hold_time_s=dwell,
                        target=observed["target"],
                        orientation_error_deg=math.degrees(observed["error"]),
                        joint_torque=motor.applied_effort[0].cpu().tolist(),
                        joint_temperature=motor._winding_T[0].cpu().tolist(),
                        joint_position=robot.data.joint_pos[0].cpu().tolist(),
                    )
                )
                video.append_data(base.render())
                if step % 150 == 0:
                    print("DEX_PROGRESS", step, counters, flush=True)
        poses = np.asarray(frames, dtype="<f4")
        assert np.isfinite(poses).all()
        poses.tofile(a.output / "poses.bin")
        (a.output / "telemetry.json").write_text(json.dumps(samples, allow_nan=False))
        par = motor._p
        result = dict(
            precision_sequence=a.precision_sequence,
            terminal_hold=a.terminal_hold,
            finger_damping=a.finger_damping,
            stable_stages=stage_done if a.precision_sequence else None,
            complete_sequence=(
                (all(stage_done) and counters["drops"] == 0)
                if a.precision_sequence
                else None
            ),
            frames=steps,
            bodies=poses.shape[1],
            fps=1 / base.step_dt,
            duration=(steps - 1) * base.step_dt,
            physics_dt_s=cfg.sim.dt,
            task=TASK,
            label=a.label,
            checkpoint_sha256=hashlib.sha256(
                Path(a.checkpoint).read_bytes()
            ).hexdigest(),
            seed=a.seed,
            goals=counters["goals"],
            drops=counters["drops"],
            joint_names=robot.joint_names,
            peak_temperature_C=max(s["temperature"] for s in samples),
            peak_torque_Nm=max(s["torque"] for s in samples),
            Kt=par.Kt,
            Kv=45 / (math.pi * par.Kt),
            gear=par.gear_ratio,
            I_peak=par.I_peak,
            V_bus=par.V_bus,
            protocol="Learned policy on corrected v1 real-motor physics. Random orientation goals change on success. Timeouts and maximum-success termination disabled; drops reset the episode and are counted. No object-state writes after environment initialization outside the normal drop reset. One 24-second trial, not a dexterous handoff or Rubik solver. Video and replay start after the same first control step.",
        )
        if a.precision_sequence:
            result["protocol"] = (
                "Six fixed 10-second stages; 0.5s continuous hold within 0.1rad, angular speed<0.5rad/s, linear speed<0.05m/s, position error<0.12m. Final stage includes 0.08N physical push at 51s for 0.2s. All six stages and no drops required for sequence completion. This single recording is illustrative; use the matched multi-seed evaluation to compare policies."
            )
        (a.output / "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False)
        )
        print("DEX_COMPLETE", counters, flush=True)
    finally:
        env.close()


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
    raise
finally:
    timer = threading.Timer(15, lambda: os._exit(0))
    timer.daemon = True
    timer.start()
    app.close(wait_for_replicator=False)
