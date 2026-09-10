"""eval_policy.py — quantitative cross-evaluation for the A/B experiment.

Loads a trained rsl_rl checkpoint and runs it in ANY registered task (ideal or
Atlas-FOC actuators), aggregating the numbers that quantify the sim-to-real gap:

    mean episode reward, mean episode length, falls vs timeouts

Usage (on the Isaac Lab pod):
    ./isaaclab.sh -p eval_policy.py --task <TASK_ID> --ckpt <model.pt> \
        --num_envs 256 --steps 1500
Prints one line:  EVAL_RESULT {json}
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=1500)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import atlas_actuators.tasks  # noqa: F401, E402  (registers Atlas tasks)
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                        device=agent_cfg.device)
runner.load(args_cli.ckpt)
policy = runner.get_inference_policy(device=agent_cfg.device)

obs, _ = env.get_observations()
dev = obs.device
ep_rew = torch.zeros(args_cli.num_envs, device=dev)
ep_len = torch.zeros(args_cli.num_envs, device=dev)
fin_rews, fin_lens = [], []
falls = timeouts = 0

with torch.inference_mode():
    for i in range(args_cli.steps):
        obs, rew, dones, extras = env.step(policy(obs))
        ep_rew += rew
        ep_len += 1
        d = dones.bool().squeeze(-1) if dones.dim() > 1 else dones.bool()
        if d.any():
            to = extras.get("time_outs")
            to = (to.bool().squeeze(-1) if to is not None
                  else torch.zeros_like(d))[d] if to is not None else torch.zeros(int(d.sum()), dtype=torch.bool, device=dev)
            fin_rews += ep_rew[d].tolist()
            fin_lens += ep_len[d].tolist()
            timeouts += int(to.sum())
            falls += int(d.sum()) - int(to.sum())
            ep_rew[d] = 0.0
            ep_len[d] = 0.0

n = max(len(fin_rews), 1)
result = {
    "task": args_cli.task, "ckpt": args_cli.ckpt,
    "episodes": len(fin_rews),
    "mean_ep_reward": round(sum(fin_rews) / n, 2),
    "mean_ep_length": round(sum(fin_lens) / n, 1),
    "falls": falls, "timeouts": timeouts,
    "fall_rate_pct": round(100.0 * falls / max(falls + timeouts, 1), 1),
}
print("EVAL_RESULT " + json.dumps(result), flush=True)
env.close()
simulation_app.close()
