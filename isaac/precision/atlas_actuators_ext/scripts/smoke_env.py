"""smoke_env.py — first in-engine exercise of the Atlas FOC actuator.

Makes the AtlasFOC ANYmal task with a few envs, steps it with random actions,
and prints applied-torque stats. Cheap tripwire before committing to training.

    ./isaaclab.sh -p smoke_env.py --task Isaac-Velocity-Flat-Anymal-C-AtlasFOC-v0
"""
import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Flat-Anymal-C-AtlasFOC-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import atlas_actuators.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
env = gym.make(args_cli.task, cfg=env_cfg)
print(f"[smoke] env made: {args_cli.task}  obs={env.observation_space}  "
      f"act={env.action_space}", flush=True)

robot = env.unwrapped.scene["robot"]
if "Allegro" in args_cli.task:
    assert robot.num_joints == 16, f"Expected 16 Allegro joints, got {robot.num_joints}"
act_group = list(robot.actuators.values())[0]
print(f"[smoke] actuator class: {type(act_group).__name__}", flush=True)

if "GearedReal20" in args_cli.task:
    assert type(act_group).__name__ == "AtlasActuatorFOC"
    assert math.isclose(act_group._dt, env_cfg.sim.dt)
    expected_mode = "projected" if args_cli.task.endswith("-v1") else "legacy"
    assert act_group._p.pack_limit_mode == expected_mode
elif "NaivePeak20" in args_cli.task:
    assert type(act_group).__name__ == "IdealPDActuator"
obs, _ = env.reset()
tau_max_seen = 0.0
for i in range(args_cli.steps):
    action = 0.5 * torch.randn(args_cli.num_envs, env.action_space.shape[1],
                               device=env.unwrapped.device)
    obs, rew, term, trunc, extras = env.step(action)
    tau = robot.data.applied_torque
    assert torch.isfinite(tau).all(), "Nonfinite applied torque"
    assert torch.isfinite(rew).all(), "Nonfinite rewards"
    for value in obs.values() if isinstance(obs, dict) else [obs]:
        assert torch.isfinite(value).all(), "Nonfinite observations"
    if hasattr(act_group, "_winding_T"):
        assert torch.isfinite(act_group._winding_T).all(), "Nonfinite winding temperature"
    tau_max_seen = max(tau_max_seen, float(tau.abs().max()))
    if i % 20 == 0:
        print(f"[smoke] step {i}: applied τ mean={float(tau.abs().mean()):.2f} "
              f"max={float(tau.abs().max()):.2f} N·m  rew={float(rew.mean()):.3f}",
              flush=True)

print(f"[smoke] DONE — {type(act_group).__name__} ran {args_cli.steps} steps; "
      f"max |τ| seen = {tau_max_seen:.1f} N·m", flush=True)
if hasattr(act_group, "_winding_T") and not act_group.cfg.reset_thermal_on_episode:
    before_reset = act_group._winding_T.clone()
    act_group.reset()
    torch.testing.assert_close(act_group._winding_T, before_reset)
assert tau_max_seen > 0, "No applied torque during smoke test"
print(f"SMOKE_OK task={args_cli.task} steps={args_cli.steps} max_torque={tau_max_seen:.6f}", flush=True)
env.close()
simulation_app.close()
