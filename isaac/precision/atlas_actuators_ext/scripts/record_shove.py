"""record_shove.py — synchronized identical shoves, for countable comparisons.

Every robot walks forward at 1 m/s and receives the same shove (fixed direction
and magnitude, fixed interval), so two policies filmed with this script can be
compared by counting falls per wave. Staggered random shoves blur that count;
synchronized identical ones make it legible.

    ./isaaclab.sh -p record_shove.py --task <TASK> --ckpt <model.pt> \
        --out <dir> --shove 1.8 --num_envs 16 --steps 750
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--shove", type=float, default=1.8)
parser.add_argument("--shove_mode", default="fixed", choices=["fixed", "random"],
                    help="fixed: every robot shoved +x at --shove; random: "
                         "direction drawn per event from +/-shove on both axes "
                         "(matches the eval protocol; env seed fixes the draw, "
                         "so both policies face the identical sequence)")
parser.add_argument("--vx", type=float, default=1.0)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=750)
parser.add_argument("--shove_period", type=float, default=4.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import atlas_actuators.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
# vx >= 0 pins a fixed forward command; vx < 0 keeps the task's own command
# distribution, matching the eval_push protocol the reported numbers use
if args_cli.vx >= 0.0:
    rng = env_cfg.commands.base_velocity.ranges
    rng.lin_vel_x = (args_cli.vx, args_cli.vx)
    rng.lin_vel_y = (0.0, 0.0)
    rng.ang_vel_z = (0.0, 0.0)
    if hasattr(rng, "heading"):
        rng.heading = (0.0, 0.0)
    env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)

# G1's task cfg disables push randomization entirely; re-create the standard
# event so humanoid policies face the identical stressor the quadrupeds do.
if getattr(env_cfg.events, "push_robot", None) is None:
    from isaaclab.managers import EventTermCfg as EventTerm
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
    env_cfg.events.push_robot = EventTerm(
        func=mdp.push_by_setting_velocity, mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}})

# identical shove clock for every robot
s = args_cli.shove
if args_cli.shove_mode == "fixed":
    env_cfg.events.push_robot.params["velocity_range"] = {"x": (s, s), "y": (0.0, 0.0)}
else:
    env_cfg.events.push_robot.params["velocity_range"] = {"x": (-s, s), "y": (-s, s)}
env_cfg.events.push_robot.interval_range_s = (args_cli.shove_period, args_cli.shove_period)

agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env, video_folder=args_cli.out,
    step_trigger=lambda step: step == 0,
    video_length=args_cli.steps, disable_logger=True)
env = RslRlVecEnvWrapper(env)

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args_cli.ckpt)
policy = runner.get_inference_policy(device=agent_cfg.device)

obs, _ = env.get_observations()
falls = 0
timeline = []          # (frame_index, falls_this_step)
with torch.inference_mode():
    for i in range(args_cli.steps + 20):
        obs, _, dones, extras = env.step(policy(obs))
        d = dones.bool().squeeze(-1) if dones.dim() > 1 else dones.bool()
        if d.any():
            to = extras.get("time_outs")
            n_to = int(to.bool().squeeze(-1)[d].sum()) if to is not None else 0
            n_falls = int(d.sum()) - n_to
            if n_falls > 0 and i < args_cli.steps:
                falls += n_falls
                timeline.append([i, n_falls])
import json as _json
import os as _os
with open(_os.path.join(args_cli.out, "falls.json"), "w") as fh:
    _json.dump({"total": falls, "timeline": timeline, "steps": args_cli.steps}, fh)
print(f"SHOVE_VIDEO_DONE falls_in_clip={falls}", flush=True)
env.close()
simulation_app.close()
