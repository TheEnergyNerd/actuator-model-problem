"""Passive internal cube mechanics; no desired-state or command inputs."""

import torch


def local_torques(rotations, angular_velocity, detent=0.02, guide=0.2, damping=0.0001):
    relative = rotations[0].T @ rotations
    tau = detent * torch.cross(
        relative.transpose(-1, -2), relative.pow(3).transpose(-1, -2), dim=-1
    ).sum(-2)
    indices = torch.arange(len(rotations), device=rotations.device)
    flat = relative.abs().flatten(1).argmax(1)
    row = flat // 3
    col = flat % 3
    vector = relative[indices, :, col]
    axis = torch.nn.functional.one_hot(row, 3).to(rotations.dtype)
    tau += guide * relative[indices, row, col, None] * torch.cross(vector, axis, dim=-1)
    world = (rotations[0] @ tau.T).T - damping * (
        angular_velocity - angular_velocity[0]
    )
    world[0] = -world[1:].sum(0)
    return (rotations.transpose(-1, -2) @ world[..., None]).squeeze(-1)
