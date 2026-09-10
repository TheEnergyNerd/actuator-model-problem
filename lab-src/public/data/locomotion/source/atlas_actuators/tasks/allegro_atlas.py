"""allegro_atlas.py — Allegro in-hand cube reorientation with Atlas hand actuators.

Swaps the stock ideal 'fingers' actuators (0.5 N·m, instantly, at any speed,
forever) for the FOC-tier Atlas finger actuator: current-limited peak torque, a
back-EMF/voltage ceiling that costs torque at finger speed, and a grasp-hold
thermal derate. That is the real envelope a small hand motor has — the point of
the plugin, now applied to a dexterous hand.

Registered (see tasks/__init__.py):
    Isaac-Repose-Cube-Allegro-AtlasFOC-v0
    Isaac-Repose-Cube-Allegro-AtlasFOC-Play-v0
"""
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.allegro_env_cfg import (
    AllegroCubeEnvCfg,
    AllegroCubeEnvCfg_PLAY,
)

from ..foc_actuator import AtlasActuatorFOCCfg, foc_cfg_kwargs

# A finger-class Atlas design: near-direct-drive small BLDC (the FINGER motor
# from hand_actuators.py, cast to a controller dict). Peak ≈ 0.45 N·m matches
# the stock 0.5 N·m effort limit, so PEAK grip is comparable — but continuous
# is ~0.13 N·m and the winding derates as the grip is held.
CONTROLLER = {"Kt_Nm_per_A": 0.10, "Kv_rpm_per_V": 95, "R_phase_mohm": 2000,
              "L_phase_uH": 250, "pole_pairs": 7, "I_peak_A": 5.0, "I_cont_A": 1.5}
GEAR, VBUS = 1.0, 24.0


def _apply_atlas(cfg):
    kw = foc_cfg_kwargs(CONTROLLER, gear_ratio=GEAR, V_bus_V=VBUS, gear_eff=0.9)
    # a small, unventilated hand motor packed in a finger: low heat capacity and
    # poor heatsinking, so a sustained grasp reaches the derate within an episode
    kw["thermal_C_JperK"] = 6.0
    kw["thermal_R_KperW"] = 5.0
    kw["T_derate_C"] = 100.0
    cfg.scene.robot.actuators = {
        "fingers": AtlasActuatorFOCCfg(
            joint_names_expr=[".*"], stiffness=3.0, damping=0.1,
            sim_dt=cfg.sim.dt, **kw,
        )
    }


@configclass
class AllegroCubeAtlasFOCEnvCfg(AllegroCubeEnvCfg):
    """Stock Allegro repose task, FOC-tier Atlas finger actuators."""

    def __post_init__(self):
        super().__post_init__()
        _apply_atlas(self)


@configclass
class AllegroCubeAtlasFOCEnvCfg_PLAY(AllegroCubeEnvCfg_PLAY):
    """Play/eval variant (smaller scene, no obs corruption)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_atlas(self)

# ── co-design sweep: four finger actuators, one axis ─────────────────────
# The binding constraint on a hand is not peak grip, it is CONTINUOUS grip:
# in-hand manipulation is a sustained-contact task, so the winding sits near
# stall and the thermal derate decides what torque survives. F1 is the design
# above. F2-F4 are progressively larger finger motors: more continuous
# current, lower phase resistance (more copper), and more thermal mass and
# heatsinking, the way a physically bigger motor actually scales.
#
#   cand   I_peak  I_cont   R      C_th   peak N·m   continuous N·m
#   F1      5.0     1.5    2.00Ω   6.0     0.45        0.135
#   F2      7.0     3.0    1.40Ω  10.0     0.63        0.270
#   F3      9.0     4.5    1.00Ω  14.0     0.81        0.405
#   F4     11.0     6.0    0.80Ω  18.0     0.99        0.540
#
# The stock (fantasy) Allegro actuator is a flat 0.50 N·m at any speed,
# forever. Only F4 matches that continuously; the question the sweep answers
# is how much continuous rating the TASK actually needs.
CANDIDATES = {
    "F1": dict(I_peak_A=5.0,  I_cont_A=1.5, R_phase_mohm=2000, C_th=6.0,  R_th=5.0),
    "F2": dict(I_peak_A=7.0,  I_cont_A=3.0, R_phase_mohm=1400, C_th=10.0, R_th=4.2),
    "F3": dict(I_peak_A=9.0,  I_cont_A=4.5, R_phase_mohm=1000, C_th=14.0, R_th=3.6),
    "F4": dict(I_peak_A=11.0, I_cont_A=6.0, R_phase_mohm=800,  C_th=18.0, R_th=3.0),
}


def _apply_candidate(cfg, name):
    """Swap in one sweep candidate. Everything except the actuator is identical."""
    c = CANDIDATES[name]
    ctrl = dict(CONTROLLER)
    ctrl.update(I_peak_A=c["I_peak_A"], I_cont_A=c["I_cont_A"],
                R_phase_mohm=c["R_phase_mohm"])
    kw = foc_cfg_kwargs(ctrl, gear_ratio=GEAR, V_bus_V=VBUS, gear_eff=0.9)
    kw["thermal_C_JperK"] = c["C_th"]
    kw["thermal_R_KperW"] = c["R_th"]
    kw["T_derate_C"] = 100.0
    cfg.scene.robot.actuators = {
        "fingers": AtlasActuatorFOCCfg(
            joint_names_expr=[".*"], stiffness=3.0, damping=0.1,
            sim_dt=cfg.sim.dt, **kw,
        )
    }


# Four explicit classes on purpose — isaaclab's @configclass rebuilds the class
# from its annotated fields, so a parameterised factory does not survive.
@configclass
class AllegroCubeF1EnvCfg(AllegroCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _apply_candidate(self, "F1")


@configclass
class AllegroCubeF2EnvCfg(AllegroCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _apply_candidate(self, "F2")


@configclass
class AllegroCubeF3EnvCfg(AllegroCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _apply_candidate(self, "F3")


@configclass
class AllegroCubeF4EnvCfg(AllegroCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _apply_candidate(self, "F4")

# ── controls that isolate the actuator from the SOLVER ───────────────────
# The stock Allegro uses ImplicitActuatorCfg: PhysX solves the PD inside the
# solver every physics substep. AtlasActuatorFOC is an explicit ActuatorBase,
# computing one torque per control step in Python. Comparing them therefore
# changes the actuator envelope AND the integration scheme at once.
#
# IDEALPD is the missing control: an EXPLICIT actuator with the same fantasy
# flat torque limit as the stock one (0.5 N·m at any speed, forever). If it
# scores near the implicit baseline, the envelope is what matters and the F1-F4
# sweep is measuring what it claims. If it collapses too, most of the reported
# gap is a solver artefact, not physics, and must be reported as one.
@configclass
class AllegroCubeIdealPDEnvCfg(AllegroCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        from isaaclab.actuators import IdealPDActuatorCfg
        self.scene.robot.actuators = {
            "fingers": IdealPDActuatorCfg(
                joint_names_expr=[".*"], effort_limit=0.5, velocity_limit=100.0,
                stiffness=3.0, damping=0.1,
            )
        }


# F5 brackets the torque axis well past the stock actuator: if torque alone were
# the constraint, this should comfortably clear it.
CANDIDATES["F5"] = dict(I_peak_A=18.0, I_cont_A=10.0, R_phase_mohm=500,
                        C_th=30.0, R_th=2.2)


@configclass
class AllegroCubeF5EnvCfg(AllegroCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _apply_candidate(self, "F5")

# ══ THE OVERPROMISE EXPERIMENT ═══════════════════════════════════════════
# The earlier transfer test came back null because it was built backwards: the
# clamp promised 0.50 N·m and the real motor delivered 0.54 continuous / 0.99
# peak, so the policy was trained on something WEAKER than it was deployed on.
# A transfer penalty needs model > reality.
#
# The mistake that actually happens in practice is reading peak torque off a
# datasheet and using it as a flat, forever limit -- "thirty newton-meters,
# instantly, at any speed, forever". So both arms below share the SAME peak;
# they differ only in whether that peak can be SUSTAINED.
#
#   NAIVE : flat clamp at 1.875 N·m, forever            (the datasheet reading)
#   REAL  : the same finger motor geared 5:1            (0.56 cont / 1.88 peak)
#
# 5:1 also fixes the second flaw: direct drive left F1 too weak to do the task
# at all, so every arm sat on the floor. A finger is the one joint where
# reduction is nearly free -- it needs torque, not speed.
GEAR_REAL, GEAR_EFF = 5.0, 0.75
_PK = 0.10 * 5.0 * GEAR_REAL * GEAR_EFF      # 1.875 N·m peak
_CT = 0.10 * 1.5 * GEAR_REAL * GEAR_EFF      # 0.5625 N·m continuous

# Episodes must outlast the winding. tau = C*R = 6*5 = 30 s; a 20 s episode
# reaches only ~49% of steady state and reset() returns the winding to ambient,
# so the sustained derate -- the entire hand argument -- never develops.
LONG_EPISODE_S = 60.0


@configclass
class AllegroCubeNaivePeakEnvCfg(AllegroCubeEnvCfg):
    """Datasheet peak used as a flat forever-limit. What the policy is promised."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.actuators import IdealPDActuatorCfg
        self.episode_length_s = LONG_EPISODE_S
        self.scene.robot.actuators = {
            "fingers": IdealPDActuatorCfg(
                joint_names_expr=[".*"], effort_limit=_PK, velocity_limit=100.0,
                stiffness=3.0, damping=0.1,
            )
        }


@configclass
class AllegroCubeGearedRealEnvCfg(AllegroCubeEnvCfg):
    """Same motor, same peak, geared 5:1 -- but it can only sustain a third."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = LONG_EPISODE_S
        kw = foc_cfg_kwargs(CONTROLLER, gear_ratio=GEAR_REAL, V_bus_V=VBUS,
                            gear_eff=GEAR_EFF)
        # Measured last run: 60 s episodes with a per-episode thermal reset took
        # the winding to 52 C of a 100 C threshold -- 23% of the derate, so the
        # designed 3.33x overpromise was only 1.19x in practice, and the measured
        # penalty was correspondingly small. Two fixes, both physical:
        #   - heat now carries across episodes (see foc_actuator.reset)
        #   - 70 C derate onset, which is what a small unventilated finger motor
        #     packed in a hand actually has; 100 C was a big-motor number
        kw["thermal_C_JperK"] = 6.0
        kw["thermal_R_KperW"] = 5.0
        kw["T_derate_C"] = 70.0
        kw["reset_thermal_on_episode"] = False
        # ── Tier 1: the couplings a per-joint model cannot express ──────
        # Sixteen motors share one pack and one housing. Without these the
        # joints are mathematically independent, which is why saturation
        # never compounded in earlier runs.
        #   24 V / 30 A supply (720 W) for a hand + forearm, 50 mOhm pack.
        # Sag couples joints that are voltage-limited (fast); the current
        # ceiling couples them at ANY speed, which is the regime fingers
        # actually run in.
        kw["R_pack_ohm"] = 0.05
        kw["I_pack_max_A"] = 30.0
        self.scene.robot.actuators = {
            "fingers": AtlasActuatorFOCCfg(
                joint_names_expr=[".*"], stiffness=3.0, damping=0.1,
                sim_dt=self.sim.dt, **kw,
            )
        }


# ══ SECOND PASS: stock episode length, DC-correct pack, and a policy that can
# feel its windings ═══════════════════════════════════════════════════════
# The 60 s episode above was added so the derate could develop inside one
# episode. Heat now persists across episodes, so it is unnecessary, and the
# naive arm's own-world score fell from 3-6 (20 s, 0.5 N·m) to 0.2 (60 s,
# 1.875 N·m) -- the long episode was hurting both arms. Back to 20 s.
from isaaclab.managers import ObservationTermCfg as _ObsTerm
from .. import mdp as _atlas_mdp


def _real_actuators(cfg):
    kw = foc_cfg_kwargs(CONTROLLER, gear_ratio=GEAR_REAL, V_bus_V=VBUS, gear_eff=GEAR_EFF)
    kw["thermal_C_JperK"] = 6.0
    kw["thermal_R_KperW"] = 5.0
    kw["T_derate_C"] = 70.0
    kw["reset_thermal_on_episode"] = False
    kw["R_pack_ohm"] = 0.05
    kw["I_pack_max_A"] = 30.0
    cfg.scene.robot.actuators = {
        "fingers": AtlasActuatorFOCCfg(
            joint_names_expr=[".*"], stiffness=3.0, damping=0.1,
            sim_dt=cfg.sim.dt, **kw,
        )
    }


@configclass
class AllegroCubeNaivePeak20EnvCfg(AllegroCubeEnvCfg):
    """Datasheet peak as a flat forever-limit, stock 20 s episode."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.actuators import IdealPDActuatorCfg
        self.scene.robot.actuators = {
            "fingers": IdealPDActuatorCfg(
                joint_names_expr=[".*"], effort_limit=_PK, velocity_limit=100.0,
                stiffness=3.0, damping=0.1,
            )
        }


@configclass
class AllegroCubeGearedReal20EnvCfg(AllegroCubeEnvCfg):
    """Same motor geared 5:1, thermal + shared pack, stock 20 s episode."""

    def __post_init__(self):
        super().__post_init__()
        _real_actuators(self)


@configclass
class AllegroCubeGearedReal20ObsEnvCfg(AllegroCubeEnvCfg):
    """As above, and the policy observes its 16 winding temperatures.

    On the quadruped a policy that could feel its windings spread load across
    joints and fell 4x instead of 17x. Same idea for a hand: the actuator
    state is part of the state, so let the policy see it."""

    def __post_init__(self):
        super().__post_init__()
        _real_actuators(self)
        self.observations.policy.winding_T = _ObsTerm(func=_atlas_mdp.winding_temperature)


@configclass
class AllegroCubeGearedReal20V1EnvCfg(AllegroCubeGearedReal20EnvCfg):
    """Versioned correction: enforce the pack current ceiling on the same step."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators["fingers"].pack_limit_mode = "projected"


@configclass
class AllegroCubeGearedReal20ObsV1EnvCfg(AllegroCubeGearedReal20ObsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators["fingers"].pack_limit_mode = "projected"
