"""Hysteretic terminal hold around a learned rotation policy.

Only freezes commanded finger targets; physics and actuator limits remain active.
"""

import torch


class TerminalHold:
    def __init__(self, actions):
        self.latched = torch.zeros(
            actions.shape[0], dtype=torch.bool, device=actions.device
        )
        self.actions = torch.zeros_like(actions)

    def reset(self, ids=None):
        if ids is None:
            self.latched.zero_()
        else:
            self.latched[ids] = False

    def command(self, proposed, previous, error, angular_speed):
        finite = torch.isfinite(error) & torch.isfinite(angular_speed)
        self.latched &= finite & (error < 0.14)
        capture = ~self.latched & finite & (error < 0.08) & (angular_speed < 1.0)
        self.actions[capture] = previous[capture]
        self.latched |= capture
        return torch.where(self.latched[:, None], self.actions, proposed)
