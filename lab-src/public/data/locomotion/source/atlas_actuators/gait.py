"""Phase-conditioned walking rewards and directly measured G1 gait diagnostics."""

import math
import torch

PERIOD_S = 0.9


def gait_clock(env):
    steps = getattr(env, "episode_length_buf", None)
    if (
        steps is None
    ):  # Observation shape probing precedes episode-buffer initialization.
        steps = torch.zeros(env.num_envs, device=env.device)
    phase = 2 * math.pi * steps * env.step_dt / PERIOD_S
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)


def _feet(env):
    if not hasattr(env, "_atlas_gait_feet"):
        names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        env._atlas_gait_feet = (
            env.scene["robot"].find_bodies(names, preserve_order=True)[0],
            env.scene["contact_forces"].find_bodies(names, preserve_order=True)[0],
        )
    return env._atlas_gait_feet


def gait_state(env):
    body_ids, sensor_ids = _feet(env)
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    contact = sensor.data.net_forces_w[:, sensor_ids, :].norm(dim=-1) > 5.0
    z = robot.data.body_pos_w[:, body_ids, 2] - env.scene.env_origins[:, None, 2]
    speed = robot.data.body_lin_vel_w[:, body_ids, :2].norm(dim=-1)
    phase = (
        env.episode_length_buf[:, None] * env.step_dt / PERIOD_S
        + torch.tensor([0.0, 0.5], device=env.device)[None, :]
    ) % 1.0
    swing = phase < 0.4
    moving = env.command_manager.get_command("base_velocity")[:, :2].norm(dim=-1) > 0.1
    return contact, z, speed, phase, swing, moving


def contact_schedule(env):
    contact, z, speed, phase, swing, moving = gait_state(env)
    return (contact == ~swing).float().mean(-1) * moving


def swing_clearance(env):
    contact, z, speed, phase, swing, moving = gait_state(env)
    # Link-frame height target, not a claimed geometric sole clearance.
    target = 0.03 + 0.08 * torch.sin(math.pi * (phase / 0.4).clamp(0, 1))
    score = torch.exp(-((z - target) / 0.04).square())
    return (score * swing).sum(-1) * moving


def gait_metrics(env):
    contact, z, speed, phase, swing, moving = gait_state(env)
    robot = env.scene["robot"]
    body_ids, sensor_ids = _feet(env)
    landed = env.scene["contact_forces"].compute_first_contact(env.step_dt)[
        :, sensor_ids
    ]
    from .gait_diagnostics import SwingLandingCounter

    if not hasattr(env, "_atlas_swing_counter"):
        env._atlas_swing_counter = SwingLandingCounter(
            env.num_envs, env.device, env.step_dt
        )
    sustained = env._atlas_swing_counter.update(contact, env.episode_length_buf)
    return {
        "root_height_m": robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2],
        "forward_world_speed_m_s": robot.data.root_lin_vel_w[:, 0],
        "yaw_rate_abs_rad_s": robot.data.root_ang_vel_w[:, 2].abs(),
        "torso_tilt_rad": torch.acos(
            (-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1)
        ),
        "ankle_link_height_max_m": z.max(-1).values,
        "left_ankle_height_m": z[:, 0],
        "right_ankle_height_m": z[:, 1],
        "left_contact": contact[:, 0].float(),
        "right_contact": contact[:, 1].float(),
        "contact_slip_m_s": (speed * contact).sum(-1) / contact.sum(-1).clamp(min=1),
        "single_support_fraction": (contact.sum(-1) == 1).float(),
        "flight_fraction": (contact.sum(-1) == 0).float(),
        "left_touchdowns": landed[:, 0].float(),
        "right_touchdowns": landed[:, 1].float(),
        "left_sustained_swing_landings": sustained[:, 0],
        "right_sustained_swing_landings": sustained[:, 1],
        "phase_contact_match": contact_schedule(env),
    }


def flight_penalty(env):
    contact, _, _, _, _, moving = gait_state(env)
    return (~contact.any(-1)).float() * moving


def yaw_command_error(env):
    """Dense turn-rate error, including when the exponential reward saturates."""
    desired = env.command_manager.get_command("base_velocity")[:, 2]
    return (env.scene["robot"].data.root_ang_vel_w[:, 2] - desired).square()
