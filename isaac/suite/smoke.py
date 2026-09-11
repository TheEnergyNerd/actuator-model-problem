"""Boot and step a native task; this is infrastructure validation, not task success."""

import argparse
import json
import os
from pathlib import Path
import threading
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--num-envs", type=int, default=2)
parser.add_argument("--seed", type=int, default=201)
parser.add_argument(
    "--atlas", action="store_true", help="Register the local Atlas extension"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1 or args.num_envs < 1:
    parser.error("steps and num-envs must be positive")
args.output.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
result = {
    "task": args.task,
    "seed": args.seed,
    "kind": "environment_smoke",
    "controller": "zero_actions",
    "task_success": None,
    "passed": False,
    "steps_completed": 0,
    "num_envs": args.num_envs,
}
app = None
env = None
try:
    app = AppLauncher(args).app
    import gymnasium as gym
    import torch
    import isaaclab_tasks  # noqa: F401

    if args.atlas:
        import atlas_actuators.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    def finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return all(finite(v) for v in value)
        return True

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    result["physics_dt"] = cfg.sim.dt
    result["decimation"] = cfg.decimation
    (args.output / "config.txt").write_text(str(cfg.to_dict()))
    env = gym.make(args.task, cfg=cfg)
    observation, _ = env.reset(seed=args.seed)
    if not finite(observation):
        raise ValueError("Non-finite initial observation")
    result["action_shape"] = list(env.action_space.shape)
    result["episode_terminations"] = 0
    with torch.inference_mode():
        action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for step in range(args.steps):
            observation, reward, terminated, truncated, info = env.step(action)
            if not finite((observation, reward)):
                raise ValueError(f"Non-finite observation/reward at step {step}")
            result["steps_completed"] = step + 1
            result["episode_terminations"] += int((terminated | truncated).sum())
    result["passed"] = True
except Exception:
    result["error"] = traceback.format_exc()
    traceback.print_exc()
finally:
    result["wall_seconds"] = time.monotonic() - started
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("SUITE_RESULT " + json.dumps(result), flush=True)
    exit_code = 0 if result["passed"] else 1
    # Isaac shutdown can hang after successful recording; preserve the actual status.
    timer = threading.Timer(10, lambda: os._exit(exit_code))
    timer.daemon = True
    timer.start()
    if env is not None:
        env.close()
    if app is not None:
        app.close()
    raise SystemExit(exit_code)
