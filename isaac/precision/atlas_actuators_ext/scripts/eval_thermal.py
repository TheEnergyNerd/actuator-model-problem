"""eval_thermal.py — sustained-sprint thermal evaluation.

Runs a policy on the thermal task and reports, per 10 s bucket: mean forward
velocity, mean/max winding temperature, fraction of joint-steps in derate
(T > T_derate - 5), and falls. Prints one THERMAL_RESULT JSON line.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--tag", default="run")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=3000)   # 60 s at 50 Hz
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks  # noqa: F401
import atlas_actuators.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args_cli.ckpt)
policy = runner.get_inference_policy(device=agent_cfg.device)

raw = env.unwrapped
robot = raw.scene["robot"]
legs = list(robot.actuators.values())[0]
T_derate = float(legs.cfg.T_derate_C)

obs, _ = env.get_observations()
buckets = []
bucket = {"vx": 0.0, "T_mean": 0.0, "T_max": 0.0, "derate": 0.0, "n": 0, "falls": 0}
BUCKET_STEPS = 500  # 10 s
with torch.inference_mode():
    for i in range(args_cli.steps):
        obs, _, dones, extras = env.step(policy(obs))
        vx = robot.data.root_lin_vel_b[:, 0].mean().item()
        T = legs._winding_T
        bucket["vx"] += vx
        bucket["T_mean"] += T.mean().item()
        bucket["T_max"] = max(bucket["T_max"], T.max().item())
        bucket["derate"] += (T > (T_derate - 5.0)).float().mean().item()
        bucket["n"] += 1
        d = dones.bool().squeeze(-1) if dones.dim() > 1 else dones.bool()
        if d.any():
            to = extras.get("time_outs")
            n_to = int(to.bool().squeeze(-1)[d].sum()) if to is not None else 0
            bucket["falls"] += int(d.sum()) - n_to
        if (i + 1) % BUCKET_STEPS == 0:
            n = max(bucket["n"], 1)
            buckets.append({"t_s": (i + 1) / 50.0,
                            "vx": round(bucket["vx"] / n, 3),
                            "T_mean": round(bucket["T_mean"] / n, 1),
                            "T_max": round(bucket["T_max"], 1),
                            "derate_frac": round(bucket["derate"] / n, 4),
                            "falls": bucket["falls"]})
            bucket = {"vx": 0.0, "T_mean": 0.0, "T_max": 0.0, "derate": 0.0, "n": 0, "falls": 0}

print("THERMAL_RESULT " + json.dumps({"tag": args_cli.tag, "buckets": buckets}), flush=True)
env.close()
simulation_app.close()
