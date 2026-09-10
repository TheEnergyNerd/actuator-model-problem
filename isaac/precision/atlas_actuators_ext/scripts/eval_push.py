"""eval_push.py — fall rates under harder shoves, same world, both policies."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--tag", default="run")
parser.add_argument("--push", type=float, default=1.5)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=1500)
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

# G1's task cfg disables push randomization entirely; re-create the standard
# event so humanoid policies face the identical stressor the quadrupeds do.
if getattr(env_cfg.events, "push_robot", None) is None:
    from isaaclab.managers import EventTermCfg as EventTerm
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
    env_cfg.events.push_robot = EventTerm(
        func=mdp.push_by_setting_velocity, mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}})

p = args_cli.push
env_cfg.events.push_robot.params["velocity_range"] = {"x": (-p, p), "y": (-p, p)}
env_cfg.events.push_robot.interval_range_s = (5.0, 8.0)

agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args_cli.ckpt)
policy = runner.get_inference_policy(device=agent_cfg.device)

obs, _ = env.get_observations()
falls = timeouts = 0
with torch.inference_mode():
    for i in range(args_cli.steps):
        obs, _, dones, extras = env.step(policy(obs))
        d = dones.bool().squeeze(-1) if dones.dim() > 1 else dones.bool()
        if d.any():
            to = extras.get("time_outs")
            to = to.bool().squeeze(-1)[d] if to is not None else torch.zeros(int(d.sum()), dtype=torch.bool)
            timeouts += int(to.sum()); falls += int(d.sum()) - int(to.sum())
print("PUSH_RESULT " + json.dumps({"tag": args_cli.tag, "push": p,
    "falls": falls, "timeouts": timeouts,
    "fall_rate_pct": round(100.0*falls/max(falls+timeouts,1),1)}), flush=True)
env.close(); simulation_app.close()
