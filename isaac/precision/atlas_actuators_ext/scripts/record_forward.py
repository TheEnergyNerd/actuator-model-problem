"""record_forward.py — record demo footage with a fixed forward command.

The velocity task normally draws random omnidirectional commands, which makes
for confusing footage (robots crab-walking in nine directions). This script
overrides the command ranges so every robot is told the same thing: walk
forward at a fixed speed, heading locked. The result is formation footage
that is legible to humans and directly comparable across policies.

    ./isaaclab.sh -p record_forward.py --task <TASK> --ckpt <model.pt> \
        --out /opt/videos_out/name --vx 1.0 --num_envs 9 --steps 420
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--vx", type=float, default=1.0)
parser.add_argument("--num_envs", type=int, default=9)
parser.add_argument("--steps", type=int, default=420)
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
# Everyone gets the same command: straight ahead at vx, heading locked to +x.
rng = env_cfg.commands.base_velocity.ranges
rng.lin_vel_x = (args_cli.vx, args_cli.vx)
rng.lin_vel_y = (0.0, 0.0)
rng.ang_vel_z = (0.0, 0.0)
if hasattr(rng, "heading"):
    rng.heading = (0.0, 0.0)
env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)

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
with torch.inference_mode():
    for i in range(args_cli.steps + 20):
        obs, _, _, _ = env.step(policy(obs))
print("FORWARD_VIDEO_DONE", flush=True)
env.close()
simulation_app.close()
