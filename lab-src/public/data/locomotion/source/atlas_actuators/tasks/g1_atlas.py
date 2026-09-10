"""g1_atlas.py — Unitree G1 humanoid velocity task on Atlas actuators.

Same plugin, different morphology. The stock Isaac Lab G1 task drives every
joint with an ImplicitActuator whose effort_limit is 300 N·m — roughly twice
what a real G1 knee delivers (Unitree spec: 139 N·m peak) and more than ten
times what its arm joints do. The Atlas variant replaces all three actuator
groups with FOC-tier Atlas actuators, one motor design geared three ways:

    legs  (hips, knees, torso)  gear 12:1  → ~148 N·m ceiling (matches spec)
    feet  (ankles)              gear  2:1  → ~25 N·m
    arms  (shoulders, elbows, wrists)  gear 2:1 → ~25 N·m

PD gains, armature, and every reward stay stock — the actuator model is the
only thing that changes, exactly as in the ANYmal A/B.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1Rewards,
)

from ..foc_actuator import AtlasActuatorFOCCfg, foc_cfg_kwargs
from .anymal_atlas import CONTROLLER, VBUS

GEAR_LEGS, GEAR_FEET, GEAR_ARMS = 12.0, 2.0, 2.0


@configclass
class G1FlatAtlasFOCEnvCfg(G1FlatEnvCfg):
    """Stock G1 flat-terrain velocity task, every joint on Atlas FOC physics."""

    def __post_init__(self):
        super().__post_init__()
        dt = self.sim.dt
        legs_kw = foc_cfg_kwargs(CONTROLLER, gear_ratio=GEAR_LEGS, V_bus_V=VBUS)
        feet_kw = foc_cfg_kwargs(CONTROLLER, gear_ratio=GEAR_FEET, V_bus_V=VBUS)
        arms_kw = foc_cfg_kwargs(CONTROLLER, gear_ratio=GEAR_ARMS, V_bus_V=VBUS)
        self.scene.robot.actuators = {
            "legs": AtlasActuatorFOCCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                    "torso_joint",
                ],
                stiffness={
                    ".*_hip_yaw_joint": 150.0,
                    ".*_hip_roll_joint": 150.0,
                    ".*_hip_pitch_joint": 200.0,
                    ".*_knee_joint": 200.0,
                    "torso_joint": 200.0,
                },
                damping={
                    ".*_hip_yaw_joint": 5.0,
                    ".*_hip_roll_joint": 5.0,
                    ".*_hip_pitch_joint": 5.0,
                    ".*_knee_joint": 5.0,
                    "torso_joint": 5.0,
                },
                armature={
                    ".*_hip_.*": 0.01,
                    ".*_knee_joint": 0.01,
                    "torso_joint": 0.01,
                },
                sim_dt=dt,
                **legs_kw,
            ),
            "feet": AtlasActuatorFOCCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                stiffness=20.0,
                damping=2.0,
                armature=0.01,
                sim_dt=dt,
                **feet_kw,
            ),
            "arms": AtlasActuatorFOCCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                    ".*_five_joint",
                    ".*_three_joint",
                    ".*_six_joint",
                    ".*_four_joint",
                    ".*_zero_joint",
                    ".*_one_joint",
                    ".*_two_joint",
                ],
                stiffness=40.0,
                damping=10.0,
                armature={
                    ".*_shoulder_.*": 0.01,
                    ".*_elbow_.*": 0.01,
                    ".*_five_joint": 0.001,
                    ".*_three_joint": 0.001,
                    ".*_six_joint": 0.001,
                    ".*_four_joint": 0.001,
                    ".*_zero_joint": 0.001,
                    ".*_one_joint": 0.001,
                    ".*_two_joint": 0.001,
                },
                sim_dt=dt,
                **arms_kw,
            ),
        }


@configclass
class G1FlatAtlasFOCTunedEnvCfg(G1FlatAtlasFOCEnvCfg):
    """Atlas G1 with PD gains rescaled to the actuator.

    The stock gains (150-200 stiffness) were chosen for 300 N.m implicit
    joints. Scaling leg-group stiffness by the torque ratio (~148/300) keeps
    the same proportional authority relative to the ceiling; rewards and every
    other line of the task stay stock.
    """

    def __post_init__(self):
        super().__post_init__()
        legs = self.scene.robot.actuators["legs"]
        legs.stiffness = {
            ".*_hip_yaw_joint": 75.0,
            ".*_hip_roll_joint": 75.0,
            ".*_hip_pitch_joint": 100.0,
            ".*_knee_joint": 100.0,
            "torso_joint": 100.0,
        }
        legs.damping = {
            ".*_hip_yaw_joint": 3.0,
            ".*_hip_roll_joint": 3.0,
            ".*_hip_pitch_joint": 3.0,
            ".*_knee_joint": 3.0,
            "torso_joint": 3.0,
        }


# ── upright variants: posture-shaped reward, applied to BOTH worlds ──
# The stock G1 flat reward is satisfied by a crouched gait (nothing in it
# prices standing height or torso attitude). These variants add an explicit
# posture term identically in the stock-actuator and Atlas-actuator worlds,
# so gait comparisons stay apples-to-apples. Every A/B number in the docs
# uses the stock reward; the upright pair exists for gait demonstrations.


@configclass
class G1UprightRewards(G1Rewards):
    """G1Rewards plus a standing-height penalty."""

    base_height: RewTerm = RewTerm(
        func=mdp.base_height_l2,
        weight=-100.0,
        params={"target_height": 0.78},
    )


def _shape_posture(cfg):
    # posture only: standing height (class field) + torso attitude. The stride
    # (feet_air_time) term stays at the stock task's values: amplifying it let
    # the fantasy-torque policy farm the bonus by holding a leg mid-air.
    cfg.rewards.flat_orientation_l2.weight = -2.0


@configclass
class G1FlatUprightEnvCfg(G1FlatEnvCfg):
    """Stock 300 N.m actuators, posture-shaped reward."""

    rewards: G1UprightRewards = G1UprightRewards()

    def __post_init__(self):
        super().__post_init__()
        _shape_posture(self)


@configclass
class G1FlatAtlasFOCUprightEnvCfg(G1FlatAtlasFOCEnvCfg):
    """Atlas FOC actuators, identical posture-shaped reward."""

    rewards: G1UprightRewards = G1UprightRewards()

    def __post_init__(self):
        super().__post_init__()
        _shape_posture(self)


@configclass
class G1FlatAtlasFOCUprightHardEnvCfg(G1FlatAtlasFOCUprightEnvCfg):
    """Upright pair + disturbance curriculum.

    The stock G1 task ships with push randomization disabled; this variant
    re-creates the standard push event (the same stressor eval_push applies)
    so the policy trains against shoves instead of only meeting them at eval.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
        )


@configclass
class G1FlatAtlasFOCUprightHWEnvCfg(G1FlatAtlasFOCUprightEnvCfg):
    """Upright pair + hardware-effects sensing: encoder quantization and
    gear backlash on every actuator group (14-bit motor-shaft encoder,
    0.008 rad joint-side play)."""

    def __post_init__(self):
        super().__post_init__()
        for group in self.scene.robot.actuators.values():
            group.encoder_counts = 16384
            group.backlash_rad = 0.008


@configclass
class G1FlatAtlasFOCExplicitEnvCfg(G1FlatAtlasFOCEnvCfg):
    """G1 with arm damping retuned for the explicit FOC integration.

    Stock implicit-drive damping (10 N m s/rad) is inappropriate for the
    small finger armatures at a 5 ms explicit step. This configuration keeps
    motor parameters, stiffness and leg/ankle gains unchanged, and lowers
    shoulder/elbow damping to 2 and finger damping to 0.2. This is a controller
    retune, not evidence that the motor matches commercial G1 hardware.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators["arms"].damping = explicit_arm_damping()


def explicit_arm_damping():
    return {
        ".*_shoulder_.*": 2.0,
        ".*_elbow_.*": 2.0,
        ".*_five_joint": 0.2,
        ".*_three_joint": 0.2,
        ".*_six_joint": 0.2,
        ".*_four_joint": 0.2,
        ".*_zero_joint": 0.2,
        ".*_one_joint": 0.2,
        ".*_two_joint": 0.2,
    }


from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ObservationsCfg,
)
from .. import gait


@configclass
class G1GaitObservations(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        gait_clock = ObsTerm(func=gait.gait_clock)

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1WalkingRewards(G1Rewards):
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-60.0, params={"target_height": 0.76}
    )
    contact_schedule = RewTerm(func=gait.contact_schedule, weight=3.0)
    swing_clearance = RewTerm(func=gait.swing_clearance, weight=1.5)


@configclass
class G1FlatAtlasWalkingEnvCfg(G1FlatAtlasFOCExplicitEnvCfg):
    """Experimental phase-conditioned gait; requires policy adaptation/training."""

    rewards: G1WalkingRewards = G1WalkingRewards()
    observations: G1GaitObservations = G1GaitObservations()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.0025
        self.decimation = 8
        self.sim.render_interval = 8
        self.scene.contact_forces.update_period = self.sim.dt
        for actuator in self.scene.robot.actuators.values():
            actuator.sim_dt = self.sim.dt
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.flat_orientation_l2.weight = -4.0
        self.rewards.feet_slide.weight = -1.0
        self.rewards.feet_air_time.weight = 0.3
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.events.base_external_force_torque = None


@configclass
class G1GroundedWalkingRewards(G1WalkingRewards):
    flight = RewTerm(func=gait.flight_penalty, weight=-4.0)


@configclass
class G1FlatAtlasGroundedWalkingEnvCfg(G1FlatAtlasWalkingEnvCfg):
    """Walking refinement: discourage hopping and excess vertical body motion."""

    rewards: G1GroundedWalkingRewards = G1GroundedWalkingRewards()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.lin_vel_z_l2.weight = -2.0


@configclass
class G1DirectedWalkingRewards(G1GroundedWalkingRewards):
    yaw_command_error = RewTerm(func=gait.yaw_command_error, weight=-1.0)


@configclass
class G1FlatAtlasDirectedWalkingEnvCfg(G1FlatAtlasGroundedWalkingEnvCfg):
    """Preserve commanded direction while learning the phase-conditioned gait."""

    rewards: G1DirectedWalkingRewards = G1DirectedWalkingRewards()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.action_rate_l2.weight = -0.03
