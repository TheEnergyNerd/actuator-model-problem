"""Evaluate native Factory policy checkpoints; retain every episode outcome."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import zipfile
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--task", choices=["PegInsert", "GearMesh", "NutThread"], default="PegInsert"
)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--num-envs", type=int, default=16)
p.add_argument("--episodes", type=int, default=3)
p.add_argument("--seed", type=int, default=201)
p.add_argument(
    "--push-force",
    type=float,
    default=0,
    help="Held-asset local Y force in N, first episode at 0.4–0.7 s",
)
p.add_argument(
    "--record",
    action="store_true",
    help="Record env 0, episode 0 before automatic reset",
)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if not 0 <= a.push_force <= 3:
    p.error("Push force must be in [0, 3] N")
if a.num_envs < 1 or a.episodes < 1:
    p.error("Positive environments and episodes required")
with zipfile.ZipFile(a.checkpoint) as archive:
    if archive.testzip() is not None:
        p.error(
            "Invalid checkpoint; use snapshot_checkpoint.py after the training write finishes"
        )
a.output.mkdir(parents=True, exist_ok=False)
app = AppLauncher(a).app
import gymnasium as gym
import numpy as np
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry

env = None
code = 1
try:
    task = f"Isaac-Factory-{a.task}-Direct-v0"
    cfg = parse_env_cfg(task, device=a.device, num_envs=a.num_envs)
    cfg.seed = a.seed
    agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
    base = gym.make(task, cfg=cfg)
    raw = base.unwrapped
    env = RlGamesVecEnvWrapper(base, a.device, float("inf"), 1.0)
    vecenv.register(
        "AtlasFactoryEvaluation",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(
            config_name, num_actors, **kwargs
        ),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "AtlasFactoryEvaluation", "env_creator": lambda **kwargs: env},
    )
    agent_cfg["params"]["config"]["num_actors"] = a.num_envs
    agent_cfg["params"]["config"]["device"] = a.device
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(a.checkpoint)
    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.restore(str(a.checkpoint))
    player.reset()
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    player.get_batch_size(obs, 1)
    if player.is_rnn:
        player.init_rnn()
    from factory_recording import FactoryRecording

    recorder = FactoryRecording(a.output, raw) if a.record else None
    original_rewards = raw._get_rewards
    snapshots = []
    applied_push = 0.0

    def measured_rewards():
        reward = original_rewards()
        passed = raw._get_curr_successes(
            success_threshold=raw.cfg_task.success_threshold,
            check_rot=raw.cfg_task.name == "nut_thread",
        )
        snapshots.append(
            {
                "push_force_n": applied_push,
                "success": passed.detach().cpu().tolist(),
                "keypoint_error_m": raw.keypoint_dist.detach().cpu().tolist(),
                "held_position": raw.held_pos.detach().cpu().tolist(),
                "fixed_position": raw.fixed_pos.detach().cpu().tolist(),
                "peak_joint_torque_nm": raw._robot.data.applied_torque.detach()
                .abs()
                .amax(-1)
                .cpu()
                .tolist(),
            }
        )
        if recorder is not None and completed[0] == 0:
            recorder.capture(snapshots[-1])
        return reward

    raw._get_rewards = measured_rewards
    hold = np.zeros(a.num_envs)
    completed = np.zeros(a.num_envs, dtype=int)
    trials = []
    with torch.inference_mode():
        for step in range(raw.max_episode_length * a.episodes + 1):
            applied_push = a.push_force if 0.4 <= step * raw.step_dt < 0.7 else 0.0
            force = torch.zeros((a.num_envs, 1, 3), device=a.device)
            force[:, :, 1] = applied_push
            raw._held_asset.set_external_force_and_torque(
                force, torch.zeros_like(force), body_ids=[0]
            )
            actions = player.get_action(player.obs_to_torch(obs), is_deterministic=True)
            if not torch.isfinite(actions).all():
                raise ValueError("Non-finite policy action")
            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]
            row = snapshots[-1]
            success = np.array(row["success"], dtype=bool)
            hold = np.where(success, hold + raw.step_dt, 0)
            row["t"] = (step + 1) * raw.step_dt
            row["done"] = dones.cpu().tolist()
            for i, done in enumerate(row["done"]):
                if done and completed[i] < a.episodes:
                    trials.append(
                        dict(
                            environment=i,
                            episode=int(completed[i]),
                            native_final_success=bool(success[i]),
                            sustained_final_success=bool(hold[i] >= 0.5 - 1e-8),
                            final_hold_s=float(hold[i]),
                            keypoint_error_m=row["keypoint_error_m"][i],
                        )
                    )
                    completed[i] += 1
                    hold[i] = 0
            if player.is_rnn and player.states is not None:
                for state in player.states:
                    state[:, dones.bool(), :] = 0
            if (completed >= a.episodes).all():
                break
    result = dict(
        task=task,
        perturbation=dict(
            force_n=a.push_force,
            start_s=0.4,
            duration_s=0.3,
            frame="Held-asset local Y",
            application="Physical external force, first episode only",
        ),
        seed=a.seed,
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        num_envs=a.num_envs,
        episodes=a.episodes,
        completed=bool((completed >= a.episodes).all()),
        trials=trials,
        successes=sum(t["sustained_final_success"] for t in trials),
        native_successes=sum(t["native_final_success"] for t in trials),
        criterion="Native task success continuously true for the final 0.5 seconds, measured before automatic reset",
        reset_policy="Native task resets and prepared initial grasp retained",
    )
    if recorder is not None:
        recorder.finish(result)
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (a.output / "telemetry.json").write_text(
        json.dumps(snapshots, allow_nan=False) + "\n"
    )
    (a.output / "config.txt").write_text(str(cfg.to_dict()))
    print("FACTORY_RESULT " + json.dumps(result), flush=True)
    code = 0 if result["completed"] else 1
finally:
    timer = threading.Timer(5, lambda: os._exit(code))
    timer.daemon = True
    timer.start()
    if env is not None:
        env.close()
    app.close()
