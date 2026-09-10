"""Evaluation-only contact timing; filters very short force-threshold chatter."""

import torch


class SwingLandingCounter:
    """Count landings after >=0.12 s continuous air and >=0.04 s support.

    These are sustained swing/contact events, not a naturalness score. Contact
    is supplied by the evaluator's force threshold. State resets per episode.
    """

    def __init__(self, num_envs, device, dt, min_air_s=0.12, min_contact_s=0.04):
        self.dt = dt
        self.min_air_s = min_air_s
        self.min_contact_s = min_contact_s
        self.air = torch.zeros(num_envs, 2, device=device)
        self.contact = torch.zeros_like(self.air)
        self.landed_air = torch.zeros_like(self.air)
        self.emitted = torch.zeros_like(self.air, dtype=torch.bool)
        self.previous_step = torch.full(
            (num_envs,), -1, device=device, dtype=torch.long
        )

    def update(self, contact, episode_steps):
        reset = episode_steps <= self.previous_step
        self.air[reset] = 0
        self.contact[reset] = 0
        self.landed_air[reset] = 0
        self.emitted[reset] = False
        first_contact = contact & (self.contact == 0)
        self.landed_air = torch.where(first_contact, self.air, self.landed_air)
        self.air = torch.where(contact, 0.0, self.air + self.dt)
        self.contact = torch.where(contact, self.contact + self.dt, 0.0)
        self.emitted &= contact
        event = (
            contact
            & ~self.emitted
            & (self.landed_air >= self.min_air_s - 1e-6)
            & (self.contact >= self.min_contact_s - 1e-6)
        )
        self.emitted |= event
        self.previous_step.copy_(episode_steps)
        return event.float()
