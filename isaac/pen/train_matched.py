"""Equal-budget PPO fine-tuning of a supplied policy under two actuator laws.

Own training objective; upstream training/reward code is not distributed. This is
not a reproduction of the upstream training run. See TRAINING.md for protocol.
"""

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path


def install_objective(env, model, output, temperature_range=(25, 100), seed=43001000):
    import torch
    from motor_model import attach

    attach(env, output, model)
    env.cfg.episode_length_s = 12.0
    env.hold_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    original_reset, original_dones = env._reset_idx, env._get_dones
    generator = torch.Generator(device=env.device).manual_seed(seed)

    def reset(self, ids):
        original_reset(ids)
        self.hold_steps[ids] = 0
        # Draw identically in both conditions, including the ideal negative control.
        temperature = temperature_range[0] + (
            temperature_range[1] - temperature_range[0]
        ) * torch.rand(len(ids), device=self.device, generator=generator)
        for key in ("id", "iq", "energy"):
            self.motor_state[key][ids] = 0
        self.motor_state["temperature"][ids] = temperature[:, None]

    def dones(self):
        previous = self.best.clone()
        _, timeout = original_dones()
        pos, _, axis = self._pen_state()
        per, palm = self._finger_forces()
        rel = pos - self.ref_pos
        drop = (
            (rel[:, 2] < -0.03)
            | (rel[:, :2].norm(dim=-1) > 0.06)
            | (axis[:, 2].abs() > math.sin(math.pi / 4))
        )
        support = (per > 0.01).any(dim=-1)
        omega = self.pen.data.root_ang_vel_w[:, 2].abs()
        holding = self.holding & support & (omega < 0.15) & ~drop
        self.hold_steps = torch.where(holding, self.hold_steps + 1, 0)
        success = self.hold_steps >= 60
        gain = (self.best - previous).clamp(0, 4 / 60)
        self.training_reward = (
            10 * gain * support
            + 0.2 * holding
            - 0.01
            - 0.02 * self.action.square().mean(-1)
            - 0.02 * self.pen.data.root_lin_vel_w.square().sum(-1)
            - 0.02 * palm.clamp(max=2)
            - 0.02 * axis[:, 2].square()
            - 20 * drop
            + 10 * success
        )
        self.training_stats = {"success": success, "drop": drop, "timeout": timeout}
        return drop | success, timeout

    env._reset_idx = types.MethodType(reset, env)
    env._get_dones = types.MethodType(dones, env)
    env._get_rewards = types.MethodType(lambda self: self.training_reward, env)
    env._reset_idx(torch.arange(env.num_envs, device=env.device))
    env.sim.step(render=False)
    env.scene.update(dt=0)


def train(env, policy, args, ck):
    import copy
    import torch
    from torch.distributions import Normal
    from tensordict import TensorDict

    torch.manual_seed(args.train_seed)
    install_objective(
        env,
        args.model,
        args.out,
        (args.temperature_min, args.temperature_max),
        args.train_seed + 1000,
    )
    teacher = copy.deepcopy(policy).eval()
    # Preserve both observation normalizers; adapt the actor and critic only.
    policy.eval()
    actor_parameters = list(policy.actor.parameters())
    critic_parameters = list(policy.critic.parameters())
    optimizer = torch.optim.Adam(
        [
            {"params": actor_parameters, "lr": 2e-5},
            {"params": critic_parameters, "lr": 3e-4},
        ]
    )
    # Fixed small exploration avoids destroying the initial dexterous skill.
    std = 0.20
    obs = env._get_observations()["policy"]
    n = env.num_envs

    def td(x):
        return TensorDict({"policy": x}, batch_size=[x.shape[0]])

    start = time.monotonic()
    logs = []
    for iteration in range(args.iterations):
        ob, ac, lp, val, rew, done = [], [], [], [], [], []
        successes = drops = episodes = 0
        for step in range(args.rollout_steps):
            with torch.no_grad():
                mean = policy.act_inference(td(obs))
                distribution = Normal(mean, std)
                action = distribution.sample()
                value = policy.evaluate(td(obs)).squeeze(-1)
                logp = distribution.log_prob(action).sum(-1)
                next_obs, reward, terminated, timeout, _ = env.step(action)
            ob.append(obs.clone())
            ac.append(action)
            lp.append(logp)
            val.append(value)
            rew.append(reward.clone())
            done.append(terminated | timeout)
            successes += int(env.training_stats["success"].sum())
            drops += int(env.training_stats["drop"].sum())
            episodes += int((terminated | timeout).sum())
            obs = next_obs["policy"]
        rewards, dones, values = torch.stack(rew), torch.stack(done), torch.stack(val)
        with torch.no_grad():
            next_value = policy.evaluate(td(obs)).squeeze(-1)
            advantage = torch.zeros_like(rewards)
            gae = torch.zeros(n, device=env.device)
            for step in reversed(range(args.rollout_steps)):
                live = (~dones[step]).float()
                delta = rewards[step] + 0.99 * next_value * live - values[step]
                gae = delta + 0.99 * 0.95 * live * gae
                advantage[step] = gae
                next_value = values[step]
            returns = (advantage + values).flatten()
            adv = advantage.flatten()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        observations = torch.stack(ob).flatten(0, 1)
        actions = torch.stack(ac).flatten(0, 1)
        old_logp = torch.stack(lp).flatten()
        loss_value = 0.0
        for epoch in range(3):
            for ids in torch.randperm(len(adv), device=env.device).split(1024):
                mean = policy.act_inference(td(observations[ids]))
                dist = Normal(mean, std)
                ratio = (dist.log_prob(actions[ids]).sum(-1) - old_logp[ids]).exp()
                clipped = torch.minimum(
                    ratio * adv[ids], ratio.clamp(0.9, 1.1) * adv[ids]
                )
                value = policy.evaluate(td(observations[ids])).squeeze(-1)
                with torch.no_grad():
                    anchor = teacher.act_inference(td(observations[ids]))
                loss = (
                    -clipped.mean()
                    + 0.5 * (value - returns[ids]).square().mean()
                    + 0.02 * (mean - anchor).square().mean()
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    actor_parameters + critic_parameters, 1.0
                )
                optimizer.step()
                loss_value = float(loss.detach())
        entry = dict(
            iteration=iteration + 1,
            transitions=(iteration + 1) * n * args.rollout_steps,
            reward_mean=float(rewards.mean()),
            successes=successes,
            drops=drops,
            episodes=episodes,
            loss=loss_value,
            elapsed_s=time.monotonic() - start,
        )
        logs.append(entry)
        print("TRAIN " + json.dumps(entry), flush=True)
        (args.out / "training.json").write_text(json.dumps(logs, indent=2))
        if (iteration + 1) % 16 == 0 or iteration + 1 == args.iterations:
            torch.save(
                dict(
                    model_state_dict=policy.state_dict(),
                    iter=ck["iter"],
                    atlas_training_iterations=iteration + 1,
                ),
                args.out / f"policy-{iteration+1:04d}.pt",
            )
    (args.out / "complete.json").write_text(json.dumps(logs[-1], indent=2))


def main():
    from run_reference import prepare

    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", choices=["ideal", "motor"], required=True)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--iterations", type=int, default=64)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--train-seed", type=int, default=43000000)
    p.add_argument("--temperature-min", type=float, default=25)
    p.add_argument("--temperature-max", type=float, default=100)
    p.add_argument("--test-seed0", type=int, default=42000000)
    p.add_argument("--test-trials", type=int, default=64)
    args, rest = p.parse_known_args()
    if not (0 <= args.temperature_min <= args.temperature_max <= 150):
        p.error("Invalid temperature range")
    source, pen = prepare(
        args.reference.resolve(), args.out.resolve(), model=args.model
    )
    # Use only the pinned evaluator's seeded native grasp and checkpoint setup.
    source = source.split("from intervention import apply_intervention")[0]
    (args.out / "protocol.json").write_text(
        json.dumps(
            dict(
                model=args.model,
                num_envs=args.num_envs,
                iterations=args.iterations,
                rollout_steps=args.rollout_steps,
                seed=args.train_seed,
                method="Atlas PPO fine-tuning; own reward; frozen normalization; common pretrained initialization",
                training_temperature_C=[args.temperature_min, args.temperature_max],
                selection="Final fixed-budget checkpoint; no selection on test outcomes",
                test_seed0=args.test_seed0,
                test_trials=args.test_trials,
            ),
            indent=2,
        )
    )
    sys.argv = [
        str(pen / "policy/evaluate.py"),
        "--ckpt",
        str(pen / "checkpoints/best_policy.pt"),
        "--run_cfg",
        str(pen / "checkpoints/env_cfg.json"),
        "--out",
        str(args.out.resolve()),
        "--trials",
        str(args.num_envs),
        "--seed0",
        str(args.train_seed),
        "--seconds",
        "12",
        *rest,
    ]
    scope = {"__name__": "__main__", "__file__": str(pen / "policy/evaluate.py")}
    exec(compile(source, str(pen / "policy/evaluate.py"), "exec"), scope)
    if scope["pending"].any():
        raise RuntimeError("Training bank contains invalid grasps")
    try:
        train(scope["env"], scope["policy"], args, scope["ck"])
    finally:
        import os, threading

        timer = threading.Timer(
            10, lambda: os._exit(0 if (args.out / "complete.json").exists() else 1)
        )
        timer.daemon = True
        timer.start()
        scope["app"].close()


if __name__ == "__main__":
    main()
