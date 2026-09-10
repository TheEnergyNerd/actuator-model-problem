"""train.py — PPO on ANYmal-C velocity, ideal vs Atlas actuators.

The comparison IS the pitch: the reward/behaviour delta between --actuator
ideal and --actuator atlas-foc is the sim-to-real gap the plugin closes.

Run inside an Isaac Lab env (e.g. the NGC isaac-sim image + isaaclab):
    /isaac-sim/python.sh scripts/train.py --actuator atlas-foc --num_envs 4096
    /isaac-sim/python.sh scripts/train.py --actuator ideal     --num_envs 4096

This mirrors Isaac Lab's own rsl_rl train script; it registers our tasks by
importing atlas_actuators.tasks, then defers to the standard runner.
"""
import argparse
import json
from pathlib import Path
import importlib.metadata

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--actuator", default="atlas-foc",
                    choices=["ideal", "atlas", "atlas-foc"])
parser.add_argument("--task", default=None,
                    help="explicit registered task id (overrides --actuator)")
parser.add_argument("--exp_name", default=None,
                    help="experiment/log dir name (defaults to anymal_<actuator>)")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", default=None,
                    help="checkpoint to load before training (continue a run)")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402  (registers the stock tasks)
import atlas_actuators.tasks  # noqa: F401, E402  (registers the Atlas tasks)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

TASK = args.task or {"ideal": "Isaac-Velocity-Flat-Anymal-C-v0",
        "atlas": "Isaac-Velocity-Flat-Anymal-C-Atlas-v0",
        "atlas-foc": "Isaac-Velocity-Flat-Anymal-C-AtlasFOC-v0"}[args.actuator]

env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
agent_cfg.max_iterations = args.max_iterations
agent_cfg.seed = args.seed
env_cfg.seed = args.seed
agent_cfg.experiment_name = args.exp_name or f"anymal_{args.actuator}"

log_dir = Path("logs") / agent_cfg.experiment_name
log_dir.mkdir(parents=True, exist_ok=True)
versions = {}
for package in ("torch", "isaaclab", "rsl-rl-lib", "gymnasium"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = "unknown"
(log_dir / "run_config.json").write_text(json.dumps({
    "task": TASK, "arguments": vars(args), "versions": versions,
    "environment": env_cfg.to_dict(), "agent": agent_cfg.to_dict(),
}, default=str, indent=2) + "\n")
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg))
runner = OnPolicyRunner(env, agent_cfg.to_dict(),
                        log_dir=str(log_dir),
                        device=agent_cfg.device)
if args.resume:
    runner.load(args.resume)
    print(f"[train] resumed from {args.resume}", flush=True)
print(f"[train] task={TASK}  envs={args.num_envs}  iters={args.max_iterations}",
      flush=True)
runner.learn(num_learning_iterations=agent_cfg.max_iterations,
             init_at_random_ep_len=True)
env.close()
simulation_app.close()
