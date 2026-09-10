"""Precision reorientation: sustained stable holds rather than transient hits."""

import torch
from ..precision_metrics import stable_hold_mask, advance_hold
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.math import quat_error_magnitude, quat_mul, quat_from_angle_axis
from isaaclab_tasks.manager_based.manipulation.inhand.mdp.commands.orientation_command import (
    InHandReOrientationCommand,
)
from isaaclab_tasks.manager_based.manipulation.inhand.mdp.commands.commands_cfg import (
    InHandReOrientationCommandCfg,
)
from .allegro_atlas import AllegroCubeGearedReal20V1EnvCfg


def conditions(env):
    term = env.command_manager.get_term("object_pose")
    obj = env.scene["object"]
    error = quat_error_magnitude(obj.data.root_quat_w, term.quat_command_w)
    angular = obj.data.root_ang_vel_w.norm(dim=-1)
    linear = obj.data.root_lin_vel_w.norm(dim=-1)
    position = (obj.data.root_pos_w - term.pos_command_w).norm(dim=-1)
    good = stable_hold_mask(error, angular, linear, position)
    return error, angular, linear, position, good


class PrecisionCommand(InHandReOrientationCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.rotation_limit = torch.full((self.num_envs,), 0.175, device=self.device)
        self.hold_time = torch.zeros(self.num_envs, device=self.device)
        self.completed = torch.zeros(self.num_envs, device=self.device)
        self.just_completed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.metrics["stable_holds"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hold_fraction"] = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids):
        # Curriculum changes only desired orientations, never object state.
        n = len(env_ids)
        if not n:
            return
        progress = min(1.0, self._env.common_step_counter / (24 * 600))
        angle_limit = 0.35 + (torch.pi - 0.35) * progress
        axes = torch.randn((n, 3), device=self.device)
        axes /= axes.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        limit = self.rotation_limit[env_ids] if self.cfg.adaptive else angle_limit
        angles = (2 * torch.rand(n, device=self.device) - 1) * limit
        quat = quat_mul(
            quat_from_angle_axis(angles, axes), self.object.data.root_quat_w[env_ids]
        )
        self.quat_command_w[env_ids] = quat
        self.hold_time[env_ids] = 0
        self.just_completed[env_ids] = False

    def _update_metrics(self):
        error, angular, linear, position, good = conditions(self._env)
        updated, crossed = advance_hold(self.hold_time, good, self._env.step_dt)
        self.hold_time.copy_(updated)
        self.just_completed.copy_(crossed)
        self.completed += self.just_completed.float()
        self.metrics["orientation_error"] = error
        self.metrics["position_error"] = position
        self.metrics["consecutive_success"].copy_(self.completed)
        self.metrics["stable_holds"].copy_(self.completed)
        self.metrics["hold_fraction"] = good.float()

    def _update_command(self):
        if self.cfg.update_goal_on_success:
            ids = self.just_completed.nonzero(as_tuple=False).squeeze(-1)
            if len(ids):
                if self.cfg.adaptive:
                    self.rotation_limit[ids] = (
                        self.rotation_limit[ids] + 0.0872665
                    ).clamp(max=torch.pi)
                self._resample(ids)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        result = super().reset(env_ids)
        self.hold_time[env_ids] = 0
        self.completed[env_ids] = 0
        self.just_completed[env_ids] = False
        return result


@configclass
class PrecisionCommandCfg(InHandReOrientationCommandCfg):
    class_type: type = PrecisionCommand
    adaptive: bool = False


def stable_reward(env):
    e, w, v, p, good = conditions(env)
    return (
        torch.exp(-(e / 0.25).square())
        * torch.exp(-(w / 0.6).square())
        * torch.exp(-(v / 0.08).square())
        * (p < 0.12)
    )


def completion_reward(env):
    *_, good = conditions(env)
    term = env.command_manager.get_term("object_pose")
    return (good & (term.hold_time + env.step_dt >= 0.5 - 1e-6)).float()


def overshoot_penalty(env):
    e, w, _, _, _ = conditions(env)
    return torch.exp(-(e / 0.4).square()) * w.square().clamp(max=25)


def torque_gap_penalty(env):
    a = env.scene["robot"].actuators["fingers"]
    return (
        ((a.computed_effort - a.applied_effort) / 1.875).square().clamp(max=4).mean(-1)
    )


@configclass
class AllegroPrecisionEnvCfg(AllegroCubeGearedReal20V1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        old = self.commands.object_pose
        self.commands.object_pose = PrecisionCommandCfg(
            asset_name=old.asset_name,
            init_pos_offset=old.init_pos_offset,
            make_quat_unique=old.make_quat_unique,
            orientation_success_threshold=0.1,
            update_goal_on_success=True,
            debug_vis=False,
        )
        self.terminations.max_consecutive_success = None
        self.rewards.success_bonus = RewTerm(func=completion_reward, weight=250.0)
        self.rewards.stable_hold = RewTerm(func=stable_reward, weight=12.0)
        self.rewards.overshoot = RewTerm(func=overshoot_penalty, weight=-0.3)
        self.rewards.torque_gap = RewTerm(func=torque_gap_penalty, weight=-0.2)
        self.rewards.action_rate_l2.weight = -0.05
