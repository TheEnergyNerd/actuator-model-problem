"""foc_core.py — vectorized FOC-tier actuator physics for Isaac Lab.

The envelope tier (isaac/atlas_actuator.py) clamps torque with a resistive
voltage model that is OPTIMISTIC near base speed (we measured 13.75 N·m where
the true dq answer is 8.5). This module is the FOC-correct replacement, in two
fidelities, both pure elementwise ops on numpy OR torch tensors of shape
[N_envs × N_joints] — so the same physics that was validated scalar
(foc_motor.py: power balance 0.00%, sim==firmware 6e-7) runs batched on GPU.

  • foc_refs(...)          — the field-weakening reference law: torque command →
                             (id_ref, iq_ref) inside the voltage ellipse +
                             current circle + thermal derate. Identical
                             equations to foc_motor.foc_control's ref stage.
  • step_current_lag(...)  — fast path: first-order closed-loop current lag
                             toward the refs (exact for any dt via 1−e^{−ω_c·dt})
                             + torque (PM + reluctance) + thermal RC update.
                             At Isaac rates (200 Hz vs 2 kHz loop) the lag is
                             nearly converged per step — the big win is the
                             CORRECT voltage/FW ceiling.
  • step_dq(...)           — validation path: substep the true dq ODEs + PI
                             (the exact foc_control law), batched.

Parameters map from the same design controller dict as everything else.
"""
from __future__ import annotations

import math
from numbers import Real
from dataclasses import dataclass


# ── backend shim: same code on numpy and torch ───────────────────────
def _xp(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return torch
    except Exception:
        pass
    import numpy as np
    return np


def _clip(x, lo, hi):
    xp = _xp(x)
    return xp.clamp(x, lo, hi) if hasattr(xp, "clamp") else xp.clip(x, lo, hi)


def _floor_parameter(value, floor):
    """Preserve scalar arithmetic while allowing [environments, 1] designs."""
    return max(value, floor) if isinstance(value, Real) else _clip(value, floor, float("inf"))


def _maximum_parameter(a, b):
    return max(a, b) if isinstance(a, Real) and isinstance(b, Real) else _where(a > b, a, b)


def _where(c, a, b):
    return _xp(c).where(c, a, b)


# ── parameters (one set per actuator group; scalars broadcast) ───────
@dataclass
class FOCActuatorParams:
    pole_pairs: int = 20
    Rs: float = 0.03            # phase resistance (Ω)
    Ld: float = 85e-6           # d-axis inductance (H)
    Lq: float = 85e-6           # q-axis inductance (H)
    lambda_m: float = 0.00833   # PM flux linkage (Wb);  Kt = 1.5·p·λ
    V_bus: float = 48.0
    I_peak: float = 55.0        # peak current (A)
    I_cont: float = 28.0        # continuous current (A) — thermal floor
    gear_ratio: float = 9.0
    gear_eff: float = 0.90
    current_bw_hz: float = 2000.0   # closed current-loop bandwidth
    # winding thermal RC (same convention as the envelope tier)
    thermal_R_KperW: float = 1.8
    thermal_C_JperK: float = 80.0
    T_ambient_C: float = 25.0
    T_derate_C: float = 120.0
    # ── temperature coupling ────────────────────────────────────────
    # Copper resistivity rises and magnet remanence falls with temperature.
    # Both push the same way: a hot motor needs more current for the same
    # torque AND dissipates more per amp, so heating is self-reinforcing.
    # Holding Rs and lambda_m constant understates loss by ~65% at 110 C.
    T_ref_C: float = 25.0           # temperature the nominal Rs/lambda_m are quoted at
    alpha_cu_perC: float = 0.00393  # copper resistivity coefficient (1/C)
    alpha_pm_perC: float = -0.0012  # NdFeB remanence coefficient (1/C)
    # ── shared DC bus ───────────────────────────────────────────────
    # Every joint on a robot draws from one pack. When many demand current at
    # once the rail sags and EVERY joint's voltage ceiling drops together;
    # under regeneration it rises. This is the channel through which one
    # saturated actuator degrades its neighbours, and it cannot be expressed
    # by a per-joint model. 0.0 restores the old infinite-bus behaviour.
    R_pack_ohm: float = 0.0
    V_bus_min_frac: float = 0.5     # floor, so a solver blip cannot zero the bus
    # Pack current ceiling. Sag only couples joints that are VOLTAGE limited,
    # i.e. moving fast. A supply also has a hard current ceiling, and that
    # couples joints at ANY speed -- which is the regime slow, high-torque
    # joints (fingers, a loaded knee) actually live in. 0 = unlimited.
    I_pack_max_A: float = 0.0
    pack_limit_mode: str = "legacy"  # legacy reproduction or same-step projected limit

    @property
    def Kt(self) -> float:
        return 1.5 * self.pole_pairs * self.lambda_m

    @property
    def v_max(self) -> float:
        return self.V_bus / math.sqrt(3.0)


def _Rs_at(p, T):
    """Phase resistance at winding temperature T."""
    return p.Rs * (1.0 + p.alpha_cu_perC * (T - p.T_ref_C))


def _lambda_at(p, T):
    """PM flux linkage at magnet temperature T (tracked by the winding node)."""
    return p.lambda_m * (1.0 + p.alpha_pm_perC * (T - p.T_ref_C))


def foc_actuator_params(controller: dict, gear_ratio: float = 9.0,
                        gear_eff: float = 0.90, V_bus_V: float = 48.0,
                        current_bw_hz: float = 2000.0) -> FOCActuatorParams:
    """Build FOC parameters using peak-dq current and phase-peak back EMF.

    Accept Kt_Nm_per_A or explicitly named Kv_phase_peak_rpm_per_V. If both
    are supplied they must agree. The ambiguous legacy Kv_rpm_per_V remains
    metadata; existing Kt-based designs retain their historical interpretation.
    A winding change also requires appropriate resistance/inductance values;
    use design_study.parameter_batch for the fixed-copper-volume rewind model.
    """
    kv = controller.get("Kv_phase_peak_rpm_per_V")
    kt_input = controller.get("Kt_Nm_per_A")
    if kv is not None:
        kv = float(kv)
        if not math.isfinite(kv) or kv <= 0:
            raise ValueError("Kv_phase_peak_rpm_per_V must be finite and positive")
        kt_from_kv = 45.0 / (math.pi * kv)
        if kt_input is not None and not math.isclose(float(kt_input), kt_from_kv, rel_tol=1e-3):
            raise ValueError("Kt and phase-peak Kv disagree: expected Kv * Kt = 45 / pi")
        Kt = kt_from_kv if kt_input is None else float(kt_input)
    elif kt_input is not None:
        Kt = float(kt_input)
    else:
        raise ValueError("Provide Kt_Nm_per_A or Kv_phase_peak_rpm_per_V with an explicit convention")
    if not math.isfinite(Kt) or Kt <= 0:
        raise ValueError("Kt_Nm_per_A must be finite and positive")
    p = int(controller.get("pole_pairs", 20))
    R = float(controller.get("R_phase_mohm", 30.0)) * 1e-3
    L = (float(controller["L_phase_uH"]) * 1e-6 if "L_phase_uH" in controller
         else float(controller.get("L_phase_H", max(R * 5e-4, 5e-5))))
    if "Ld_H" in controller and "Lq_H" in controller:
        Ld, Lq = float(controller["Ld_H"]), float(controller["Lq_H"])
    else:
        # analytical saliency split (same law as dq_inductance.py, inlined so
        # the package is self-contained): d-axis flux crosses the magnets,
        # q-axis stays in iron for IPM → Lq/Ld > 1; SPM ≈ 1.08.
        topo = str(controller.get("topology", "outrunner")).lower()
        t_mag = float(controller.get("magnet_thickness_mm", 2.5))
        airgap = float(controller.get("airgap_mm", 0.5))
        if any(k in topo for k in ("ipm", "interior", "spoke")):
            ratio = min(max(1.0 + 0.35 * (t_mag / 1.05) / max(airgap, 1e-3), 1.2), 3.0)
        else:
            ratio = 1.08
        Ld = 2.0 * L / (1.0 + ratio)
        Lq = ratio * Ld
    return FOCActuatorParams(
        pole_pairs=p, Rs=R, Ld=Ld, Lq=Lq, lambda_m=Kt / (1.5 * p),
        V_bus=V_bus_V, I_peak=float(controller.get("I_peak_A", 55.0)),
        I_cont=float(controller.get("I_cont_A", 28.0)),
        gear_ratio=gear_ratio, gear_eff=gear_eff, current_bw_hz=current_bw_hz)


# ── the field-weakening reference law (batched) ──────────────────────
def foc_refs(p: FOCActuatorParams, tau_motor_des, omega_m, winding_T, v_bus=None,
              i_scale=None):
    """Torque command → (id_ref, iq_ref), elementwise on arrays/tensors.

    Same equations as foc_motor.foc_control's reference stage: flux budget
    from the voltage limit → negative id (field weakening) when needed →
    current-circle clamp, with the circle radius thermally derated
    I_peak→I_cont exactly like the envelope tier.
    """
    xp = _xp(tau_motor_des)
    eps = 1e-9
    omega_e = p.pole_pairs * omega_m

    # thermal derate of the current circle (I_peak → I_cont as winding heats)
    frac = _clip((winding_T - p.T_ambient_C) /
                 max(p.T_derate_C - p.T_ambient_C, 1.0), 0.0, 1.0)
    I_max_eff = p.I_peak - frac * (p.I_peak - p.I_cont)
    # the pack cannot serve every joint at once: shrink everyone's circle
    if i_scale is not None:
        I_max_eff = I_max_eff * i_scale

    # temperature-dependent magnet flux -> temperature-dependent Kt. A hot
    # motor needs MORE current for the same torque.
    lam_T = _lambda_at(p, winding_T)
    Kt_T = 1.5 * p.pole_pairs * lam_T

    # 1) clamp the current request FIRST — the flux budget must be evaluated
    #    on a feasible iq, not the raw (possibly huge) torque request.
    iq_ref = _clip(tau_motor_des / _clip(Kt_T, eps, float("inf")),
                   -I_max_eff, I_max_eff)

    # 2) voltage budget with resistive margin (0.95·vmax − R·Imax): keeps the
    #    refs inverter-feasible so the plant can actually track them (same law
    #    as foc_motor.foc_control / the emitted firmware).
    # v_max follows the ACTUAL rail: a sagging bus lowers every joint's
    # voltage ceiling at once (see R_pack_ohm).
    v_max_eff = (p.v_max if v_bus is None else v_bus / math.sqrt(3.0))
    # Clamp at zero: the resistive margin can exceed the available volts once
    # the rail sags (or at high speed), and a negative budget would propagate
    # into a negative torque ceiling. No volts left simply means no torque
    # available at this speed, not torque in reverse.
    v_budget = _clip(0.95 * v_max_eff - _Rs_at(p, winding_T) * p.I_peak,
                     0.0, float("inf"))
    fb = v_budget / _clip(abs(omega_e), 1e-6, float("inf"))      # flux budget (Wb)
    psi_q = p.Lq * iq_ref
    psi_d_max = xp.sqrt(_clip(fb * fb - psi_q * psi_q, 0.0, float("inf")))
    id_ref = _clip((psi_d_max - lam_T) / p.Ld, -float("inf"), 0.0)

    over = psi_q * psi_q > fb * fb                # q-flux alone exceeds budget
    iq_cap = fb / _floor_parameter(p.Lq, eps)
    iq_ref = _where(over, xp.sign(iq_ref) * iq_cap, iq_ref)
    id_ref = _where(over, -lam_T / p.Ld + xp.zeros_like(id_ref), id_ref)

    mag = xp.sqrt(id_ref * id_ref + iq_ref * iq_ref) + eps
    scale = _clip(I_max_eff / mag, 0.0, 1.0)
    return id_ref * scale, iq_ref * scale


def _torque_motor(p: FOCActuatorParams, id_, iq_, winding_T=None):
    """Electromagnetic torque, PM + reluctance (same formula as foc_motor).

    The PM term uses the temperature-derated flux: the same current makes
    less torque in a hot motor."""
    lam = p.lambda_m if winding_T is None else _lambda_at(p, winding_T)
    return 1.5 * p.pole_pairs * (lam * iq_ + (p.Ld - p.Lq) * id_ * iq_)


def bus_state(p: FOCActuatorParams, id_, iq_, winding_T, omega_m):
    """(v_bus, i_scale) for the shared supply — both broadcast over joints.

    i_scale < 1 means the pack cannot serve everyone at once and every joint's
    current circle shrinks by the same factor, which is what a supply hitting
    its ceiling actually does.
    """
    v = dc_bus(p, id_, iq_, winding_T, omega_m)
    if p.I_pack_max_A <= 0.0:
        return v, None
    # The pack sees DC current, which is electrical power over bus voltage.
    # The inverter is a buck stage: at low speed the phase current is large but
    # the duty cycle is small, so DC draw is several times smaller than phase
    # current. Counting phase current here (the previous version) over-drew the
    # pack by 3-5x at finger speeds. Regeneration does not count against the
    # ceiling, so per-joint draw is clamped at zero.
    Rs_T = _Rs_at(p, winding_T)
    p_cu = 1.5 * (id_ * id_ + iq_ * iq_) * Rs_T
    p_mech = _torque_motor(p, id_, iq_, winding_T) * omega_m
    i_dc = _clip((p_cu + p_mech) / _floor_parameter(p.V_bus, 1e-6), 0.0, float("inf"))
    i_tot = i_dc.sum(-1, keepdims=True) if hasattr(i_dc, "sum") else i_dc
    scale = _clip(p.I_pack_max_A / _clip(i_tot, 1e-6, float("inf")), 0.0, 1.0)
    return v, scale


def project_pack_current(p, id_, iq_, winding_T, omega_m):
    """Conservatively enforce the DC draw ceiling on this step's currents.

    With a common current scale k, copper loss is quadratic and PM mechanical
    power is linear. Treat positive reluctance power as linear too (an upper
    bound for 0 <= k <= 1), and do not credit regeneration against consumption.
    Solve A*k**2 + B*k <= I_limit*V_loaded analytically, avoiding the legacy
    delayed clamp's 50 A / 18 A oscillation on a 30 A supply at hand stall.
    """
    if p.I_pack_max_A <= 0:
        return id_, iq_
    xp = _xp(id_)
    copper = 1.5 * _Rs_at(p, winding_T) * (id_ ** 2 + iq_ ** 2)
    pm = 1.5 * p.pole_pairs * _lambda_at(p, winding_T) * iq_ * omega_m
    reluctance = 1.5 * p.pole_pairs * (p.Ld - p.Lq) * id_ * iq_ * omega_m
    A = copper.sum(-1, keepdims=True)
    B = (_clip(pm, 0.0, float("inf")) + _clip(reluctance, 0.0, float("inf"))).sum(-1, keepdims=True)
    # Loaded voltage at the ceiling is conservative for lower positive draws.
    v_floor = _maximum_parameter(p.V_bus * p.V_bus_min_frac, p.V_bus - p.R_pack_ohm * p.I_pack_max_A)
    budget = p.I_pack_max_A * v_floor
    denominator = B + xp.sqrt(B * B + 4.0 * A * budget)
    scale = _clip(2.0 * budget / _clip(denominator, 1e-9, float("inf")), 0.0, 1.0)
    return id_ * scale, iq_ * scale


def dc_bus(p: FOCActuatorParams, id_, iq_, winding_T, omega_m):
    """Per-ENV bus voltage after the shared pack's IR drop.

    Sums electrical power across the joint axis, converts to DC-side current,
    and drops it across R_pack. Positive draw sags the rail; regeneration
    (negative power, e.g. a decelerating joint) raises it. Returns a value
    broadcastable back over joints, so one joint's demand moves every other
    joint's ceiling -- the coupling a per-joint model cannot represent."""
    if p.R_pack_ohm <= 0.0:
        return None
    xp = _xp(id_)
    Rs_T = _Rs_at(p, winding_T)
    # copper loss + mechanical power, per joint
    p_cu = 1.5 * (id_ * id_ + iq_ * iq_) * Rs_T
    p_mech = _torque_motor(p, id_, iq_, winding_T) * omega_m
    p_joint = p_cu + p_mech
    p_tot = p_joint.sum(-1, keepdims=True) if hasattr(p_joint, "sum") else p_joint
    i_dc = p_tot / _floor_parameter(p.V_bus, 1e-6)
    v = p.V_bus - i_dc * p.R_pack_ohm
    return _clip(v, p.V_bus * p.V_bus_min_frac, p.V_bus * 1.5)


def _thermal_step(p: FOCActuatorParams, id_, iq_, winding_T, dt: float):
    # resistance rises with temperature -> loss rises -> more heating
    P_cu = 1.5 * (id_ * id_ + iq_ * iq_) * _Rs_at(p, winding_T)
    P_cool = (winding_T - p.T_ambient_C) / _floor_parameter(p.thermal_R_KperW, 1e-6)
    return winding_T + (P_cu - P_cool) * dt / _floor_parameter(p.thermal_C_JperK, 1e-6)


# ── fast path: quasi-static refs + first-order current lag ───────────
def step_current_lag(p: FOCActuatorParams, tau_joint_des, joint_vel,
                     id_state, iq_state, winding_T, dt: float):
    """One physics step of the FOC-tier actuator (fast path).

    Args are arrays/tensors broadcastable to [N_envs × N_joints]; returns
    (applied_joint_torque, id_state, iq_state, winding_T).
    """
    G, eta = p.gear_ratio, p.gear_eff
    tau_m_des = tau_joint_des / (G * eta)
    omega_m = joint_vel * G

    # Bus sag is evaluated on the PREVIOUS step's currents: an explicit lag of
    # one control step, which avoids an implicit solve and is what a real
    # supply does anyway (the rail responds after the load appears).
    v_bus, i_scale = bus_state(p, id_state, iq_state, winding_T, omega_m)

    if p.pack_limit_mode == "projected":
        i_scale = None  # use the current step's feasible output, not last step's load
    elif p.pack_limit_mode != "legacy":
        raise ValueError(f"Unknown pack limit mode: {p.pack_limit_mode}")
    id_ref, iq_ref = foc_refs(p, tau_m_des, omega_m, winding_T, v_bus, i_scale)
    # exact first-order closed-loop response for this dt
    alpha = 1.0 - math.exp(-2.0 * math.pi * p.current_bw_hz * dt)
    id_new = id_state + (id_ref - id_state) * alpha
    iq_new = iq_state + (iq_ref - iq_state) * alpha

    if p.pack_limit_mode == "projected":
        id_new, iq_new = project_pack_current(p, id_new, iq_new, winding_T, omega_m)
    tau_m = _torque_motor(p, id_new, iq_new, winding_T)
    winding_T = _thermal_step(p, id_new, iq_new, winding_T, dt)
    return tau_m * G * eta, id_new, iq_new, winding_T


# ── validation path: true dq ODE + PI, substepped and batched ────────
def step_dq(p: FOCActuatorParams, tau_joint_des, joint_vel,
            id_state, iq_state, int_d, int_q, winding_T,
            dt: float, substeps: int = 20):
    """Substep the exact dq dynamics + current PI (the validated foc_control
    law) inside one physics step. Costs `substeps`× the fast path — use for
    small-env validation runs, not 4096-env training."""
    if p.pack_limit_mode != "legacy":
        raise ValueError("Projected pack limiting is supported by lag mode only")
    xp = _xp(id_state)
    G, eta = p.gear_ratio, p.gear_eff
    tau_m_des = tau_joint_des / (G * eta)
    omega_m = joint_vel * G
    omega_e = p.pole_pairs * omega_m
    kp = p.Ld * 2 * math.pi * p.current_bw_hz
    ki = p.Rs * 2 * math.pi * p.current_bw_hz
    h = dt / substeps

    v_bus, i_scale = bus_state(p, id_state, iq_state, winding_T, omega_m)
    id_ref, iq_ref = foc_refs(p, tau_m_des, omega_m, winding_T, v_bus, i_scale)
    Rs_T = _Rs_at(p, winding_T)
    lam_T = _lambda_at(p, winding_T)
    v_max = p.v_max if v_bus is None else v_bus / math.sqrt(3.0)
    for _ in range(substeps):
        ed, eq = id_ref - id_state, iq_ref - iq_state
        int_d = int_d + ed * h
        int_q = int_q + eq * h
        vd = kp * ed + ki * int_d - omega_e * p.Lq * iq_state
        vq = kp * eq + ki * int_q + omega_e * (p.Ld * id_state + lam_T)
        vmag = xp.sqrt(vd * vd + vq * vq) + 1e-9
        s = _clip(v_max / vmag, 0.0, 1.0)
        sat = s < 1.0
        vd, vq = vd * s, vq * s
        int_d = _where(sat, int_d - ed * h, int_d)     # anti-windup
        int_q = _where(sat, int_q - eq * h, int_q)
        did = (vd - Rs_T * id_state + omega_e * p.Lq * iq_state) / p.Ld
        diq = (vq - Rs_T * iq_state - omega_e * (p.Ld * id_state + lam_T)) / p.Lq
        id_state = id_state + did * h
        iq_state = iq_state + diq * h

    tau_m = _torque_motor(p, id_state, iq_state, winding_T)
    winding_T = _thermal_step(p, id_state, iq_state, winding_T, dt)
    return tau_m * G * eta, id_state, iq_state, int_d, int_q, winding_T


def sense_encoder_backlash(joint_pos, motor_pos, meas_prev, desired_effort,
                           encoder_counts, gear_ratio, backlash_rad, dt,
                           traverse_speed=None):
    """Hardware-effects sensing model: gear backlash + encoder quantization.

    Models the two measurement artifacts a motor-shaft encoder cannot escape:

    - Backlash (play operator): the motor-side position `motor_pos` (expressed
      joint-side) is dragged by the joint only through gear-face contact, so it
      holds still while the joint back-drives through the gap. Torque only
      transmits when the motor presses on the face matching its sign; on a
      torque reversal the motor crosses the gap (one control step of zero
      transmission) before re-engaging.
    - Encoder quantization: position is measured on the motor shaft with
      `encoder_counts` per revolution, i.e. a joint-side step of
      2*pi/(counts*gear); velocity is the finite difference of that quantized
      signal.

    All tensors are (num_envs, num_joints). Returns
    (meas_pos, meas_vel, transmitted_scale, motor_pos, meas_prev).
    Pass encoder_counts=0 / backlash_rad=0.0 to disable either effect.
    """
    import torch

    if backlash_rad > 0.0:
        half = 0.5 * backlash_rad
        # joint drags motor through gear-face contact (play operator)
        motor_pos = torch.clamp(motor_pos, joint_pos - half, joint_pos + half)
        # motor seeks the face its torque presses on. Traversal happens at the
        # motor's free speed, so torque transmits for the fraction of the step
        # spent in contact rather than gating to zero for the whole step.
        want_hi = desired_effort > 0.0
        eps = 1e-3 * half
        on_hi = (motor_pos - joint_pos) >= (half - eps)
        on_lo = (joint_pos - motor_pos) >= (half - eps)
        engaged = torch.where(want_hi, on_hi, on_lo)
        target_face = torch.where(want_hi, joint_pos + half, joint_pos - half)
        if traverse_speed is None:
            traverse_speed = backlash_rad / dt  # legacy: crosses in one step
        gap = (target_face - motor_pos).abs()
        t_cross = gap / max(traverse_speed, 1e-9)
        frac_dead = torch.clamp(t_cross / dt, 0.0, 1.0)
        reach = traverse_speed * dt
        moved = torch.clamp(target_face - motor_pos, -reach, reach)
        motor_pos = torch.where(engaged, motor_pos, motor_pos + moved)
        transmitted = torch.where(engaged, torch.ones_like(gap), 1.0 - frac_dead)
    else:
        motor_pos = joint_pos
        transmitted = torch.ones_like(joint_pos)

    if encoder_counts > 0:
        step = (2.0 * 3.141592653589793) / (encoder_counts * gear_ratio)
        meas_pos = torch.round(motor_pos / step) * step
    else:
        meas_pos = motor_pos
    meas_vel = (meas_pos - meas_prev) / dt
    return meas_pos, meas_vel, transmitted, motor_pos, meas_pos
