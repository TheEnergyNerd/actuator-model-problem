"""Shared strict grasp-hold criterion, independent of the simulator runtime."""

import torch


def stable_hold_mask(error, angular_speed, linear_speed, position_error):
    return (
        (error < 0.1)
        & (angular_speed < 0.5)
        & (linear_speed < 0.05)
        & (position_error < 0.12)
    )


def advance_hold(elapsed, qualified, dt, required=0.5):
    """Accumulate uninterrupted valid time and flag only the first crossing."""
    updated = torch.where(qualified, elapsed + dt, torch.zeros_like(elapsed))
    crossed = (elapsed < required - 1e-6) & (updated >= required - 1e-6)
    return updated, crossed


def bounded_goal_step(current, target, rate, dt):
    """Rate-limit a desired orientation. Does not touch the simulated object."""
    dot = (current * target).sum(-1, keepdim=True)
    target = torch.where(dot < 0, -target, target)
    theta = torch.acos(dot.abs().clamp(max=1.0))
    fraction = (rate * dt / (2 * theta).clamp_min(1e-8)).clamp(max=1.0)
    sine = torch.sin(theta).clamp_min(1e-8)
    blend = (
        torch.sin((1 - fraction) * theta) * current
        + torch.sin(fraction * theta) * target
    ) / sine
    blend = torch.where(theta < 1e-5, target, blend)
    return blend / blend.norm(dim=-1, keepdim=True).clamp_min(1e-8)
