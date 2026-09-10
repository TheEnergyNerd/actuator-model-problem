"""Fine-tune an existing real-motor policy for sustained precision holds."""

import argparse, json, hashlib, os, threading, traceback
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--iterations", type=int, default=600)
p.add_argument("--num_envs", type=int, default=2048)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--eval-only", action="store_true")
p.add_argument("--terminal-hold", action="store_true")
p.add_argument("--eval-seeds", default="101,102,103")
p.add_argument("--eval-envs", type=int, default=32)
p.add_argument("--physics-substeps", type=int, default=1)
p.add_argument("--initial-std", type=float, default=0.25)
p.add_argument("--adaptive-curriculum", action="store_true")
p.add_argument("--entropy-coef", type=float, default=0.001)
p.add_argument("--goal-rate", type=float, default=0.0)
p.add_argument("--eval-seconds", type=float, default=60.0)
p.add_argument("--finger-damping", type=float, default=None)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.output.exists():
    p.error("Output exists")
if a.physics_substeps < 1 or a.iterations < 1 or min(a.num_envs, a.eval_envs) < 1:
    p.error("Substeps, iterations, and environment counts must be positive")
if not 0 < a.eval_seconds <= 60:
    p.error("Evaluation duration must be in (0, 60] seconds")
if a.finger_damping is not None and a.finger_damping < 0:
    p.error("Finger damping must be nonnegative")
if a.terminal_hold and not a.eval_only:
    p.error("Terminal hold is currently an evaluation-only controller")
a.headless = True
app = AppLauncher(a).app
import gymnasium as gym, torch, numpy as np
import isaaclab_tasks, atlas_actuators.tasks
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.math import quat_mul, quat_from_angle_axis, quat_error_magnitude

TASK = "Isaac-Repose-Cube-Allegro-Precision-v1"


def main():
    a.output.mkdir(parents=True)
    import shutil

    shutil.copy2(__file__, a.output / "runner_at_launch.py")
    import atlas_actuators.tasks.allegro_precision as task_module

    shutil.copy2(task_module.__file__, a.output / "task_at_launch.py")
    cfg = parse_env_cfg(TASK, num_envs=a.eval_envs if a.eval_only else a.num_envs)
    cfg.seed = a.seed
    cfg.commands.object_pose.adaptive = a.adaptive_curriculum
    cfg.sim.dt /= a.physics_substeps
    cfg.decimation *= a.physics_substeps
    cfg.sim.render_interval = cfg.decimation
    for actuator in cfg.scene.robot.actuators.values():
        actuator.sim_dt = cfg.sim.dt
        if a.finger_damping is not None:
            actuator.damping = a.finger_damping
    if a.eval_only:
        cfg.observations.policy.enable_corruption = False
        cfg.terminations.time_out = None
        cfg.commands.object_pose.update_goal_on_success = False
    agent = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent.seed = a.seed
    agent.algorithm.learning_rate = 1e-4
    agent.algorithm.schedule = "fixed"
    agent.algorithm.entropy_coef = a.entropy_coef
    agent.save_interval = 100
    (a.output / "config.json").write_text(
        json.dumps(
            dict(
                arguments=vars(a),
                checkpoint_sha256=hashlib.sha256(
                    Path(a.checkpoint).read_bytes()
                ).hexdigest(),
                env=cfg.to_dict(),
                agent=agent.to_dict(),
            ),
            default=str,
            indent=2,
        )
    )
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg), clip_actions=agent.clip_actions)
    try:
        runner = OnPolicyRunner(
            env, agent.to_dict(), log_dir=str(a.output), device=agent.device
        )
        runner.load(a.checkpoint, load_optimizer=False)
        if not a.eval_only:
            with torch.no_grad():
                runner.alg.policy.std.fill_(a.initial_std)
            runner.learn(
                num_learning_iterations=a.iterations, init_at_random_ep_len=True
            )
            runner.save(str(a.output / "final.pt"))
        else:
            base = env.unwrapped
            policy = runner.get_inference_policy(device=base.device)
            term = base.command_manager.get_term("object_pose")
            obj = base.scene["object"]
            motor = base.scene["robot"].actuators["fingers"]
            mass = base.scene["robot"].root_physx_view.get_generalized_mass_matrices()
            mass = torch.as_tensor(mass, device=base.device)
            eig = torch.linalg.eigvalsh(mass)
            modal = torch.linalg.eigvals(
                torch.linalg.solve(mass, torch.diag_embed(motor.damping))
            ).real.amax(-1)
            inertia = dict(
                min_mass_eigenvalue=float(eig.min()),
                max_damping_eigenvalue=float(modal.max()),
                unconstrained_explicit_damping_dt_bound_s=float(2 / modal.max()),
                physics_dt_s=cfg.sim.dt,
            )
            (a.output / "inertia_diagnostic.json").write_text(
                json.dumps(inertia, indent=2)
            )
            print("INERTIA_DIAGNOSTIC", inertia, flush=True)
            results = []
            from atlas_actuators.tasks.allegro_precision import conditions

            for seed in map(int, a.eval_seeds.split(",")):
                base.seed(seed)
                env.reset()
                motor._winding_T.fill_(25.0)
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
                                torch.full((a.eval_envs,), angle, device=base.device),
                                axes[axis].repeat(a.eval_envs, 1),
                            ),
                            initial,
                        )
                    )
                planned = initial.clone()
                from atlas_actuators.hold_controller import TerminalHold

                previous_action = torch.zeros((a.eval_envs, 16), device=base.device)
                terminal = TerminalHold(previous_action)
                from atlas_actuators.precision_metrics import (
                    bounded_goal_step,
                    stable_hold_mask,
                )

                holds = torch.zeros((a.eval_envs, 6), device=base.device)
                dwell = torch.zeros(a.eval_envs, device=base.device)
                drops = torch.zeros(a.eval_envs, device=base.device)
                errsum = drops.clone()
                shortfall = drops.clone()
                peakT = drops.clone()
                rows = []
                force = torch.zeros((a.eval_envs, 1, 3), device=base.device)
                torque = torch.zeros_like(force)
                for step in range(round(a.eval_seconds / base.step_dt)):
                    t = step * base.step_dt
                    phase = min(5, int(t / 10))
                    if step % round(10 / base.step_dt) == 0:
                        dwell.zero_()
                        terminal.reset()
                    planned = (
                        bounded_goal_step(
                            planned, targets[phase], a.goal_rate, base.step_dt
                        )
                        if a.goal_rate > 0
                        else targets[phase]
                    )
                    term.quat_command_w[:] = planned
                    term.hold_time.zero_()
                    force.zero_()
                    if 51 <= t < 51.2:
                        force[:, 0, 0] = 0.08
                        holds[:, 5] = 0
                        dwell.zero_()
                    obj.set_external_force_and_torque(force, torque)
                    with torch.no_grad():
                        obs, _ = env.get_observations()
                        action = policy(obs)
                        if a.terminal_hold:
                            action = terminal.command(
                                action,
                                previous_action,
                                quat_error_magnitude(
                                    obj.data.root_quat_w, targets[phase]
                                ),
                                obj.data.root_ang_vel_w.norm(dim=-1),
                            )
                        previous_action.copy_(action)
                        _, _, done, _ = env.step(action)
                        terminal.reset(done.bool())
                    _, w, v, pos, _ = conditions(base)
                    e = quat_error_magnitude(obj.data.root_quat_w, targets[phase])
                    good = stable_hold_mask(e, w, v, pos) & ~done.bool()
                    drops += done.float()
                    dwell = torch.where(good, dwell + base.step_dt, 0.0)
                    holds[:, phase] = torch.maximum(holds[:, phase], dwell)
                    errsum += e
                    shortfall += (
                        (
                            (motor.computed_effort - motor.applied_effort).abs()
                            > 0.05 * motor.computed_effort.abs().clamp(min=0.1)
                        )
                        .float()
                        .mean(-1)
                    )
                    peakT = torch.maximum(peakT, motor._winding_T.max(-1).values)
                    if step % 150 == 0:
                        print(
                            "PRECISION_EVAL",
                            seed,
                            round(t, 1),
                            "holds",
                            int((holds >= 0.5 - 1e-6).sum()),
                            flush=True,
                        )
                steps = round(a.eval_seconds / base.step_dt)
                success = holds >= 0.5 - 1e-6
                complete = success.all(-1) & (drops == 0)
                results.append(
                    dict(
                        seed=seed,
                        stage_success=success.cpu().tolist(),
                        max_dwell_s=holds.cpu().tolist(),
                        complete=complete.cpu().tolist(),
                        drops=drops.cpu().tolist(),
                        mean_orientation_error_rad=(errsum / steps).cpu().tolist(),
                        torque_shortfall_fraction=(shortfall / steps).cpu().tolist(),
                        peak_winding_C=peakT.cpu().tolist(),
                    )
                )
                (a.output / "progress.json").write_text(
                    json.dumps({"results": results})
                )
                print(
                    "SEED_COMPLETE",
                    seed,
                    "stages",
                    int(success.sum()),
                    "sequences",
                    int(complete.sum()),
                    flush=True,
                )
            (a.output / "evaluation.json").write_text(
                json.dumps(
                    dict(
                        goal_rate_rad_s=a.goal_rate,
                        terminal_hold=a.terminal_hold,
                        evaluation_seconds=a.eval_seconds,
                        physics_dt_s=cfg.sim.dt,
                        finger_damping=a.finger_damping,
                        checkpoint_sha256=hashlib.sha256(
                            Path(a.checkpoint).read_bytes()
                        ).hexdigest(),
                        results=results,
                        protocol="Six fixed 10-second stages: initial orientation, Y45, Y90, X90, Z90, return + physical 0.08 N cube-local-x pulse at 51s for 0.2s. Success requires 0.5 continuous seconds with error<0.1 rad, angular speed<0.5 rad/s, linear speed<0.05 m/s, position error<0.12m. Complete sequence requires all six stages and zero drops. Final stage dwell restarts at pulse. 32 environments per evaluation seed; normal drop resets counted.",
                    ),
                    indent=2,
                )
            )
        (a.output / "status.json").write_text(json.dumps({"status": "completed"}))
        print("PRECISION_COMPLETE", flush=True)
    finally:
        env.close()


try:
    main()
except Exception:
    traceback.print_exc()
    if a.output.exists():
        (a.output / "status.json").write_text(json.dumps({"status": "failed"}))
    raise
finally:
    timer = threading.Timer(15, lambda: os._exit(0))
    timer.daemon = True
    timer.start()
    app.close(wait_for_replicator=False)
