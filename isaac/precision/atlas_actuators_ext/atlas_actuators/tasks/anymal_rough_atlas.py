"""Rough-terrain training entry point using the existing Atlas FOC motor model.

This is a training environment, not a parkour policy. Terrain, observations,
rewards and curriculum match the native ANYmal rough task. Motor mass is
unchanged here; mass interventions belong to explicit design experiments.
"""

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import (
    AnymalCRoughEnvCfg,
)

from ..foc_actuator import AtlasActuatorFOCCfg, foc_cfg_kwargs
from .anymal_atlas import CONTROLLER, GEAR, VBUS


@configclass
class AnymalCRoughAtlasFOCEnvCfg(AnymalCRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators = {
            "legs": AtlasActuatorFOCCfg(
                joint_names_expr=[".*"],
                stiffness=85.0,
                damping=2.0,
                sim_dt=self.sim.dt,
                **foc_cfg_kwargs(CONTROLLER, gear_ratio=GEAR, V_bus_V=VBUS),
            )
        }
