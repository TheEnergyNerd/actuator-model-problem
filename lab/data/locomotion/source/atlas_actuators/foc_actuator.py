"""foc_actuator.py — AtlasActuatorFOC: the FOC-tier actuator as an Isaac Lab plugin.

Same ActuatorBase contract as the envelope-tier AtlasActuator, but the torque
clamp is the FOC-correct one: field-weakening voltage ellipse (with resistive
margin), current circle with thermal derate, and a first-order current lag —
the batched physics in foc_core, which is test-locked to the scalar dq model
that carries the 0.00% power-balance validation and the 6e-7 firmware parity.

    from atlas_actuators import AtlasActuatorFOCCfg, foc_cfg_kwargs
    robot_cfg.actuators = {"legs": AtlasActuatorFOCCfg(
        joint_names_expr=[".*"], stiffness=120.0, damping=6.0,
        **foc_cfg_kwargs(controller, gear_ratio=9.0, V_bus_V=48.0))}

mode="lag" (default) is one elementwise pass per physics step — fine at
4096 envs. mode="dq" substeps the true dq ODE + PI (validation runs).
"""
from __future__ import annotations

from .foc_core import (FOCActuatorParams, sense_encoder_backlash,
                       step_current_lag, step_dq)

try:
    from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
    from isaaclab.utils import configclass
    import torch
    _HAVE_ISAACLAB = True
except ModuleNotFoundError as error:
    if error.name != "isaaclab":
        raise
    _HAVE_ISAACLAB = False


def foc_cfg_kwargs(controller: dict, gear_ratio: float = 9.0,
                   V_bus_V: float = 48.0, gear_eff: float = 0.90) -> dict:
    """AtlasActuatorFOCCfg kwargs from a design run's controller dict.

    Module-level on purpose: isaaclab's @configclass rebuilds config classes
    from their annotated fields and drops staticmethods (learned in-engine)."""
    from .foc_core import foc_actuator_params
    p = foc_actuator_params(controller, gear_ratio=gear_ratio,
                            gear_eff=gear_eff, V_bus_V=V_bus_V)
    return dict(pole_pairs=p.pole_pairs, Rs_ohm=p.Rs, Ld_H=p.Ld, Lq_H=p.Lq,
                lambda_m_Wb=p.lambda_m, V_bus_V=p.V_bus, I_peak_A=p.I_peak,
                I_cont_A=p.I_cont, gear_ratio=p.gear_ratio, gear_eff=p.gear_eff)


if _HAVE_ISAACLAB:

    class AtlasActuatorFOC(ActuatorBase):
        """FOC-tier Atlas actuator: PD sets a desired torque, the FOC physics
        decides what the inverter+motor can actually deliver."""

        cfg: "AtlasActuatorFOCCfg"

        def __init__(self, cfg, *args, **kwargs):
            super().__init__(cfg, *args, **kwargs)
            self._p = FOCActuatorParams(
                pole_pairs=cfg.pole_pairs, Rs=cfg.Rs_ohm, Ld=cfg.Ld_H,
                Lq=cfg.Lq_H, lambda_m=cfg.lambda_m_Wb, V_bus=cfg.V_bus_V,
                I_peak=cfg.I_peak_A, I_cont=cfg.I_cont_A,
                gear_ratio=cfg.gear_ratio, gear_eff=cfg.gear_eff,
                current_bw_hz=cfg.current_bw_hz,
                R_pack_ohm=getattr(cfg, "R_pack_ohm", 0.0),
                I_pack_max_A=getattr(cfg, "I_pack_max_A", 0.0),
                pack_limit_mode=cfg.pack_limit_mode,
                T_ref_C=getattr(cfg, "T_ref_C", 25.0),
                alpha_cu_perC=getattr(cfg, "alpha_cu_perC", 0.00393),
                alpha_pm_perC=getattr(cfg, "alpha_pm_perC", -0.0012),
                thermal_R_KperW=cfg.thermal_R_KperW,
                thermal_C_JperK=cfg.thermal_C_JperK,
                T_ambient_C=cfg.T_ambient_C, T_derate_C=cfg.T_derate_C)
            shape = (self._num_envs, self.num_joints)
            dev = self._device
            self._id = torch.zeros(shape, device=dev)
            self._iq = torch.zeros(shape, device=dev)
            self._int_d = torch.zeros(shape, device=dev)
            self._int_q = torch.zeros(shape, device=dev)
            self._winding_T = torch.full(shape, float(cfg.T_ambient_C), device=dev)
            self._dt = float(getattr(cfg, "sim_dt", 1.0 / 200.0))
            # optional learned residual on top of the physics prior
            self._res, self._res_n, self._res_i = None, 1, 0
            ck = getattr(cfg, "residual_ckpt", None)
            if ck:
                from .residual import load_residual_checkpoint
                self._res = load_residual_checkpoint(ck, device=dev)
                self._res_n = max(1, int(getattr(cfg, "residual_every_n", 1)))
                self._res_last = None
            if cfg.residual_max_Nm < 0:
                raise ValueError("residual_max_Nm must be nonnegative")
            if cfg.mode not in ("lag", "dq"):
                raise ValueError(f"Unknown actuator mode: {cfg.mode}")
            if cfg.mode == "dq" and cfg.pack_limit_mode != "legacy":
                raise ValueError("Projected pack limiting requires lag mode")
            self._mode = cfg.mode
            # hardware-effects sensing state (encoder + backlash), seeded lazily
            self._hw_on = (cfg.encoder_counts > 0) or (cfg.backlash_rad > 0.0)
            # gap-traversal speed: motor free speed mapped to the joint side
            _vmax = cfg.V_bus_V / (3.0 ** 0.5)
            self._lash_v = _vmax / (cfg.lambda_m_Wb * cfg.pole_pairs) / cfg.gear_ratio
            self._motor_pos = torch.zeros(shape, device=dev)
            self._meas_prev = torch.zeros(shape, device=dev)
            self._hw_seeded = torch.zeros(shape, dtype=torch.bool, device=dev)
            self._prev_desired = torch.zeros(shape, device=dev)

        def compute(self, control_action, joint_pos, joint_vel):
            meas_pos, meas_vel, transmitted = joint_pos, joint_vel, None
            if self._hw_on:
                fresh = ~self._hw_seeded
                if fresh.any():
                    self._motor_pos = torch.where(fresh, joint_pos, self._motor_pos)
                    self._meas_prev = torch.where(fresh, joint_pos, self._meas_prev)
                    self._hw_seeded |= True
                meas_pos, meas_vel, transmitted, self._motor_pos, self._meas_prev = \
                    sense_encoder_backlash(
                        joint_pos, self._motor_pos, self._meas_prev,
                        self._prev_desired, self.cfg.encoder_counts,
                        self.cfg.gear_ratio, self.cfg.backlash_rad, self._dt,
                        traverse_speed=self._lash_v)
            error_pos = control_action.joint_positions - meas_pos
            error_vel = control_action.joint_velocities - meas_vel
            desired = (self.stiffness * error_pos + self.damping * error_vel
                       + control_action.joint_efforts)
            if self._hw_on:
                self._prev_desired = desired
            if self._mode == "dq":
                applied, self._id, self._iq, self._int_d, self._int_q, \
                    self._winding_T = step_dq(
                        self._p, desired, joint_vel, self._id, self._iq,
                        self._int_d, self._int_q, self._winding_T,
                        self._dt, substeps=self.cfg.dq_substeps)
            else:
                applied, self._id, self._iq, self._winding_T = step_current_lag(
                    self._p, desired, joint_vel, self._id, self._iq,
                    self._winding_T, self._dt)
            if transmitted is not None:
                applied = applied * transmitted
            self.computed_effort = desired
            # ── learned residual on the physics prior ────────────────
            # Correction only: `applied` is already the full physics answer.
            # The residual is bounded and zero when untrained, so this line is
            # a no-op unless a validated checkpoint was supplied.
            if self._res is not None:
                self._res_i += 1
                if self._res_i % self._res_n == 0 or self._res_last is None:
                    with torch.no_grad():
                        feats = torch.stack([
                            error_pos, error_vel, desired, applied,
                            self._winding_T, self._id, self._iq,
                        ], dim=-1)
                        i_mag = (self._id ** 2 + self._iq ** 2).sqrt()
                        aggs = torch.stack([
                            i_mag.sum(-1),                       # SUM: bus draw
                            (1.5 * i_mag ** 2 * self._p.Rs).sum(-1),  # SUM: copper loss
                            self._winding_T.amax(-1),            # MAX: hottest winding
                            self._winding_T.mean(-1),
                        ], dim=-1)
                        correction = self._res(feats, aggs)
                        if correction.shape != applied.shape or not torch.isfinite(correction).all():
                            raise ValueError("Residual must return finite [envs, joints] torques")
                        self._res_last = correction.clamp(-self.cfg.residual_max_Nm,
                                                          self.cfg.residual_max_Nm)
                applied = applied + self._res_last
            self.applied_effort = applied
            control_action.joint_efforts = applied
            control_action.joint_positions = None
            control_action.joint_velocities = None
            return control_action

        def reset(self, env_ids=None):
            """Zero the electrical state on an episode reset.

            The winding temperature is deliberately NOT reset by default. Copper
            does not return to ambient because an episode ended: a hand doing
            repeated grasps, or a leg walking all afternoon, accumulates heat
            across attempts, and the derate that follows is the whole point of
            modelling thermal at all. Resetting it every episode caps how far
            the winding can ever climb and quietly disables the mechanism.
            Set `reset_thermal_on_episode=True` for the old behaviour.
            """
            if env_ids is None:
                env_ids = slice(None)
            for buf in (self._id, self._iq, self._int_d, self._int_q):
                buf[env_ids] = 0.0
            if getattr(self.cfg, "reset_thermal_on_episode", False):
                self._winding_T[env_ids] = float(self.cfg.T_ambient_C)
            self._hw_seeded[env_ids] = False
            self._prev_desired[env_ids] = 0.0
            # A cached correction belongs to the old episode, even if thermal
            # state persists. Refresh the batch on the next compute.
            if self._res is not None:
                self._res_last = None
                self._res_i = 0

    @configclass
    class AtlasActuatorFOCCfg(ActuatorBaseCfg):
        class_type: type = AtlasActuatorFOC
        # electrical (from the design's controller dict via foc_cfg_kwargs)
        pole_pairs: int = 20
        Rs_ohm: float = 0.03
        Ld_H: float = 85e-6
        Lq_H: float = 85e-6
        lambda_m_Wb: float = 0.00833
        V_bus_V: float = 48.0
        I_peak_A: float = 55.0
        I_cont_A: float = 28.0
        gear_ratio: float = 9.0
        gear_eff: float = 0.90
        current_bw_hz: float = 2000.0
        # thermal
        thermal_R_KperW: float = 1.8
        thermal_C_JperK: float = 80.0
        T_ambient_C: float = 25.0
        # False = winding heat carries across episodes (physical).
        reset_thermal_on_episode: bool = False
        # Shared DC bus: pack resistance seen by ALL joints on this robot.
        # >0 couples the joints -- heavy simultaneous demand sags the rail and
        # lowers every joint's torque ceiling together. 0 = infinite bus.
        R_pack_ohm: float = 0.0
        I_pack_max_A: float = 0.0   # supply current ceiling shared by all joints
        pack_limit_mode: str = "legacy"
        # ── learned residual (optional) ──────────────────────────────
        # Path to a TorchScript/state_dict JointResidual trained on measured
        # (commanded - delivered) data. None = pure physics, which is the
        # default: an unvalidated residual is worse than no residual.
        residual_ckpt: str | None = None
        residual_max_Nm: float = 0.25     # hard bound on the correction
        residual_every_n: int = 1         # evaluate every N control steps
        # Temperature coupling (copper resistivity up, magnet flux down)
        T_ref_C: float = 25.0
        alpha_cu_perC: float = 0.00393
        alpha_pm_perC: float = -0.0012
        T_derate_C: float = 120.0
        # hardware effects (0 = ideal sensing, off by default)
        encoder_counts: int = 0      # motor-shaft encoder counts per revolution
        backlash_rad: float = 0.0    # joint-side gear play, total width
        # integration
        sim_dt: float = 1.0 / 200.0
        mode: str = "lag"            # "lag" (fast, training) | "dq" (validation)
        dq_substeps: int = 20
else:
    AtlasActuatorFOC = None      # type: ignore
    AtlasActuatorFOCCfg = None   # type: ignore
