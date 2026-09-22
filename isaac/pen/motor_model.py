"""Matched explicit PD drives: constant torque versus an assumed FOC motor model.

This is a model-sensitivity experiment, not a calibration of Sharpa hardware.
Both arms of the comparison have identical PD gains, stall ceilings, geometry,
mass, inertia, observations and policy. Only the torque-delivery law differs.
"""

import json
import os
from pathlib import Path
import sys
import types
import torch

CORE = (
    Path(__file__).resolve().parents[1]
    / "precision/atlas_actuators_ext/atlas_actuators"
)
sys.path.insert(0, str(CORE))
from foc_core import FOCActuatorParams, step_current_lag


def attach(env, output, mode):
    if mode == "native":
        return
    if mode not in ("ideal", "motor", "motor-hot"):
        raise ValueError(mode)
    robot = env.robot
    actuator = robot.actuators["hand"]
    stiffness = actuator.stiffness.clone()
    damping = actuator.damping.clone()
    limits = robot.data.joint_effort_limits.clone()
    p = FOCActuatorParams(
        pole_pairs=7,
        Rs=2.0,
        Ld=200e-6,
        Lq=200e-6,
        lambda_m=0.012 / (1.5 * 7),
        V_bus=12.0,
        I_peak=2.0,
        I_cont=1.0,
        gear_ratio=limits / (0.012 * 2.0 * 0.8),
        gear_eff=0.8,
        current_bw_hz=500,
        thermal_R_KperW=10.0,
        thermal_C_JperK=8.0,
        T_ambient_C=25.0,
        T_derate_C=85.0,
        alpha_pm_perC=0.0,
    )
    profile = os.environ.get("ATLAS_MOTOR_PROFILE", "generic")
    if profile not in ("generic", "mj5208-virtual"):
        raise ValueError(f"Unknown motor profile: {profile}")
    evidence = None
    if profile == "mj5208-virtual":
        from bench_profile import electrical_parameters, provenance, KT

        for key, value in electrical_parameters().items():
            setattr(p, key, value)
        p.gear_ratio = limits / (KT * p.I_peak * p.gear_eff)
        evidence = provenance()
    # Equal fixed stall torque in the simplified law; FOC adds voltage/current
    # feasibility, electrical lag, winding resistance and thermal derating.
    robot.write_joint_stiffness_to_sim(torch.zeros_like(stiffness))
    robot.write_joint_damping_to_sim(torch.zeros_like(damping))
    robot._has_implicit_actuators = False
    actuator.stiffness = stiffness
    actuator.damping = damping
    state = dict(
        id=torch.zeros_like(limits),
        iq=torch.zeros_like(limits),
        temperature=torch.full_like(limits, 100.0 if mode == "motor-hot" else 25.0),
        energy=torch.zeros_like(limits),
    )
    env.motor_state = state

    def compute(self, action, joint_pos, joint_vel):
        desired = (
            self.stiffness * (action.joint_positions - joint_pos)
            + self.damping * (action.joint_velocities - joint_vel)
            + action.joint_efforts
        )
        request = desired.clamp(-limits, limits)
        if mode == "ideal":
            applied = request
            # An ideal drive has no electrical/thermal predictions. Keep these
            # fields absent in the public display, rather than reporting zeros.
        else:
            applied, state["id"], state["iq"], state["temperature"] = step_current_lag(
                p,
                request,
                joint_vel,
                state["id"],
                state["iq"],
                state["temperature"],
                env.physics_dt,
            )
            loss = (
                1.5
                * (state["id"] ** 2 + state["iq"] ** 2)
                * p.Rs
                * (1 + p.alpha_cu_perC * (state["temperature"] - p.T_ref_C))
            )
            state["energy"] += loss * env.physics_dt
        state.update(
            requested=request,
            applied=applied,
            velocity=joint_vel.clone(),
            motor_rpm=joint_vel * p.gear_ratio * 60 / (2 * torch.pi),
        )
        self.computed_effort = desired
        self.applied_effort = applied
        action.joint_positions = None
        action.joint_velocities = None
        action.joint_efforts = applied
        return action

    actuator.compute = types.MethodType(compute, actuator)
    config = dict(
        mode=mode,
        controller="Same explicit PD and native gains in both model conditions",
        parameter_status="Assumed generic geared motors; not measured Sharpa hardware parameters",
        phase_peak_Kv_rpm_per_V=45 / (torch.pi * 0.012),
        Kt_Nm_per_peak_q_A=0.012,
        bus_voltage_V=12.0,
        peak_current_A=2.0,
        continuous_current_A=1.0,
        phase_resistance_ohm=2.0,
        Ld_H=200e-6,
        Lq_H=200e-6,
        gear_efficiency=0.8,
        thermal_R_K_per_W=10.0,
        thermal_C_J_per_K=8.0,
        thermal_derating="Linear 2 A to 1 A from 25 C to 85 C; resistance increases with temperature",
        initial_temperature_C=100.0 if mode == "motor-hot" else 25.0,
        physics_dt=env.physics_dt,
        joint_names=robot.joint_names,
        gear_ratio=p.gear_ratio[0].cpu().tolist(),
        stall_torque_Nm=limits[0].cpu().tolist(),
        stiffness=stiffness[0].cpu().tolist(),
        damping=damping[0].cpu().tolist(),
        changes_after="Identical native prepared-grasp settling; before the first policy action",
    )
    if evidence is not None:
        config.update(
            parameter_status=evidence["status"],
            bench_evidence=evidence,
            phase_peak_Kv_rpm_per_V=evidence["derived"]["phase_peak_Kv_rpm_per_V"],
            Kt_Nm_per_peak_q_A=KT,
            phase_resistance_ohm=p.Rs,
            Ld_H=p.Ld,
            Lq_H=p.Lq,
            current_bandwidth_Hz=p.current_bw_hz,
        )
    config["profile"] = profile
    (Path(output) / "motor-model.json").write_text(json.dumps(config, indent=2))
    env.motor_mode = mode
