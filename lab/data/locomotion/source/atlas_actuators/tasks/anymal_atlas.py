"""anymal_atlas.py — example task: ANYmal-C velocity tracking with Atlas actuators.

Takes Isaac Lab's stock flat-terrain ANYmal-C velocity task and swaps the ideal
actuators for Atlas ones — that's the entire point of the plugin: train the
same policy against the actuator the hardware will actually have.

Registered ids (see tasks/__init__.py):
    Isaac-Velocity-Flat-Anymal-C-Atlas-v0        (envelope tier)
    Isaac-Velocity-Flat-Anymal-C-AtlasFOC-v0     (FOC tier)

Only importable inside an Isaac Lab python env; verified boot path is the same
NGC image the plugin was validated on.
"""
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from .. import mdp as atlas_mdp

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import (
    AnymalCFlatEnvCfg,
)

from ..foc_actuator import AtlasActuatorFOCCfg, foc_cfg_kwargs

# A quadruped-leg-class Atlas design (the one that held the ANYmal up in the
# rendered demo). Swap in any /api/actuator/design controller dict.
CONTROLLER = {"Kt_Nm_per_A": 0.25, "Kv_rpm_per_V": 38, "R_phase_mohm": 30,
              "L_phase_uH": 85, "pole_pairs": 20, "I_peak_A": 55, "I_cont_A": 28}
GEAR, VBUS = 9.0, 48.0


@configclass
class AnymalCFlatAtlasEnvCfg(AnymalCFlatEnvCfg):
    """Stock task, envelope-tier Atlas actuators."""

    def __post_init__(self):
        super().__post_init__()
        from ..envelope import AtlasActuatorCfg, actuator_cfg_kwargs
        self.scene.robot.actuators = {
            "legs": AtlasActuatorCfg(
                joint_names_expr=[".*"], stiffness=85.0, damping=2.0,
                **actuator_cfg_kwargs(CONTROLLER, gear_ratio=GEAR, V_bus_V=VBUS),
            )
        }


@configclass
class AnymalCFlatAtlasFOCEnvCfg(AnymalCFlatEnvCfg):
    """Stock task, FOC-tier Atlas actuators (field-weakening-correct ceiling)."""

    def __post_init__(self):
        super().__post_init__()
        kw = foc_cfg_kwargs(CONTROLLER, gear_ratio=GEAR, V_bus_V=VBUS)
        self.scene.robot.actuators = {
            "legs": AtlasActuatorFOCCfg(
                joint_names_expr=[".*"], stiffness=85.0, damping=2.0,
                sim_dt=self.sim.dt, **kw,
            )
        }


# ── co-design sweep variants: same motor, different gear ratio ───────
# Gear trades peak joint torque against joint speed range (base speed / G).
# Three explicit classes on purpose — isaaclab's @configclass drops
# unannotated class attributes, so no clever parameterized base.

@configclass
class AnymalCFlatAtlasFOCG6EnvCfg(AnymalCFlatEnvCfg):
    """Atlas FOC, gear 6:1 — torque-light (74 N·m peak), fast joints."""

    def __post_init__(self):
        super().__post_init__()
        kw = foc_cfg_kwargs(CONTROLLER, gear_ratio=6.0, V_bus_V=VBUS)
        self.scene.robot.actuators = {
            "legs": AtlasActuatorFOCCfg(
                joint_names_expr=[".*"], stiffness=85.0, damping=2.0,
                sim_dt=self.sim.dt, **kw,
            )
        }


@configclass
class AnymalCFlatAtlasFOCG12EnvCfg(AnymalCFlatEnvCfg):
    """Atlas FOC, gear 12:1 — torque-rich (148 N·m peak), slow joints."""

    def __post_init__(self):
        super().__post_init__()
        kw = foc_cfg_kwargs(CONTROLLER, gear_ratio=12.0, V_bus_V=VBUS)
        self.scene.robot.actuators = {
            "legs": AtlasActuatorFOCCfg(
                joint_names_expr=[".*"], stiffness=85.0, damping=2.0,
                sim_dt=self.sim.dt, **kw,
            )
        }


# ── disturbance-hardened variants: same tasks, shoves to ±2.0 in training ──
# Both variants get the identical curriculum so the actuator model remains the
# only difference between them.

@configclass
class AnymalCFlatHardPushEnvCfg(AnymalCFlatEnvCfg):
    """Stock actuators (ANYdrive net), training shoves widened to ±2.0 m/s."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.params["velocity_range"] = {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}


@configclass
class AnymalCFlatAtlasFOCHardEnvCfg(AnymalCFlatEnvCfg):
    """Atlas FOC actuators, training shoves widened to ±2.0 m/s."""

    def __post_init__(self):
        super().__post_init__()
        kw = foc_cfg_kwargs(CONTROLLER, gear_ratio=GEAR, V_bus_V=VBUS)
        self.scene.robot.actuators = {
            "legs": AtlasActuatorFOCCfg(
                joint_names_expr=[".*"], stiffness=85.0, damping=2.0,
                sim_dt=self.sim.dt, **kw,
            )
        }
        self.events.push_robot.params["velocity_range"] = {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}


# ── sprint sweep: the gear axis under a task that demands torque at speed ──
# Flat walking at 1 m/s never stresses the actuator; commanding up to
# 2.5 m/s pushes joints toward the speed range where gearing decides
# what torque survives (Fig: torque-speed).

def _make_sprint(base):
    class _Sprint(base):
        def __post_init__(self):
            super().__post_init__()
            r = self.commands.base_velocity.ranges
            r.lin_vel_x = (0.0, 2.5)
    _Sprint.__name__ = base.__name__.replace("EnvCfg", "SprintEnvCfg")
    return configclass(_Sprint)


AnymalCFlatAtlasFOCG6SprintEnvCfg = _make_sprint(AnymalCFlatAtlasFOCG6EnvCfg)
AnymalCFlatAtlasFOCSprintEnvCfg = _make_sprint(AnymalCFlatAtlasFOCEnvCfg)
AnymalCFlatAtlasFOCG12SprintEnvCfg = _make_sprint(AnymalCFlatAtlasFOCG12EnvCfg)


# ── thermal experiment: sustained sprint against a hot actuator ──
# Thermal capacitance cut 10x so a 60 s episode reaches derating; the
# ThermalObs variant lets the policy SEE its winding temperatures, the
# blind variant runs the identical physics without the sense.

@configclass
class AnymalCFlatAtlasFOCThermalEnvCfg(AnymalCFlatEnvCfg):
    """Blind thermal variant: hot actuator, no temperature observation."""

    def __post_init__(self):
        super().__post_init__()
        kw = foc_cfg_kwargs(CONTROLLER, gear_ratio=GEAR, V_bus_V=VBUS)
        # a small, unventilated actuator: fast heating, poor heatsinking, and
        # a conservative derate ceiling, so 60 s of sprinting meets the limit
        kw["thermal_C_JperK"] = 8.0
        kw["thermal_R_KperW"] = 5.0
        kw["T_derate_C"] = 85.0
        self.scene.robot.actuators = {
            "legs": AtlasActuatorFOCCfg(
                joint_names_expr=[".*"], stiffness=85.0, damping=2.0,
                sim_dt=self.sim.dt, **kw,
            )
        }
        # tau_thermal = R*C = 40 s: a 60 s episode ends at the derate line's
        # doorstep (~75 C measured); 120 s puts the back half deep in derate
        self.episode_length_s = 120.0
        r = self.commands.base_velocity.ranges
        r.lin_vel_x = (1.0, 2.5)
        r.lin_vel_y = (0.0, 0.0)
        r.ang_vel_z = (0.0, 0.0)
        # resample within the episode: gives learning a path from moderate to
        # hard commands instead of a cold-start wall at sprint speed
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)


@configclass
class AnymalCFlatAtlasFOCThermalObsEnvCfg(AnymalCFlatAtlasFOCThermalEnvCfg):
    """Identical physics; the policy also observes winding temperatures."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.winding_T = ObsTerm(func=atlas_mdp.winding_temperature)
