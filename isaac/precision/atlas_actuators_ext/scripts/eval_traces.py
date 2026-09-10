"""eval_traces.py — make the actuator-mismatch difference measurable.

Runs a policy in the FOC world under a fixed forward command and logs, per
control step: mean achieved forward velocity, and the fraction of joint-steps
where the policy's demanded torque exceeded what the actuator could deliver
(|computed_effort| - |applied_effort| > tol). The first is the tracking signal
the reward summarizes; the second is a direct count of the policy asking for
physics it does not have.

    ./isaaclab.sh -p eval_traces.py --task <TASK> --ckpt <model.pt> \
        --vx 1.0 --num_envs 64 --steps 500 --tag A_foc
Prints one line:  TRACE_RESULT {json}
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--tag", default="run")
parser.add_argument("--vx", type=float, default=1.0)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=500)
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
import atlas_actuators.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
rng = env_cfg.commands.base_velocity.ranges
rng.lin_vel_x = (args_cli.vx, args_cli.vx)
rng.lin_vel_y = (0.0, 0.0)
rng.ang_vel_z = (0.0, 0.0)
if hasattr(rng, "heading"):
    rng.heading = (0.0, 0.0)
env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)

agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args_cli.ckpt)
policy = runner.get_inference_policy(device=agent_cfg.device)

robot = env.unwrapped.scene["robot"]
act_group = list(robot.actuators.values())[0]

obs, _ = env.get_observations()
vx_trace = []
sat_total = 0
sat_steps = 0
TOL = 0.5  # N·m; demanded exceeds deliverable by more than this counts as clamped

with torch.inference_mode():
    for i in range(args_cli.steps):
        obs, _, _, _ = env.step(policy(obs))
        vx = robot.data.root_lin_vel_b[:, 0]           # body-frame forward velocity
        vx_trace.append(round(float(vx.mean()), 4))
        demanded = act_group.computed_effort.abs()
        applied = act_group.applied_effort.abs()
        sat_total += int((demanded - applied > TOL).sum())
        sat_steps += demanded.numel()

result = {
    "tag": args_cli.tag, "task": args_cli.task, "vx_cmd": args_cli.vx,
    "num_envs": args_cli.num_envs, "steps": args_cli.steps,
    "mean_vx": round(sum(vx_trace) / len(vx_trace), 3),
    "saturation_pct": round(100.0 * sat_total / max(sat_steps, 1), 2),
    "vx_trace_50": [vx_trace[i] for i in range(0, len(vx_trace), 10)],
}
print("TRACE_RESULT " + json.dumps(result), flush=True)
env.close()
simulation_app.close()
