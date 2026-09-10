"""Observation terms exposed by Atlas actuators."""
import torch

from isaaclab.managers import SceneEntityCfg


def winding_temperature(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Per-joint winding temperature from Atlas FOC actuators, normalized.

    (T - ambient) / 100 so a cold joint reads 0.0 and a 125 C winding ~1.0.
    Only actuator groups that carry a thermal state contribute.
    """
    asset = env.scene[asset_cfg.name]
    temps = []
    for group in asset.actuators.values():
        T = getattr(group, "_winding_T", None)
        if T is not None:
            temps.append(T)
    return (torch.cat(temps, dim=-1) - 25.0) / 100.0
