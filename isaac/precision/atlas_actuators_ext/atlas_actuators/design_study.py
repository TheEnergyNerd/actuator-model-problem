"""Explicit design interventions for paired, batched Isaac Lab experiments.

Kv means mechanical rpm per phase-peak back-EMF volt. With this package's
peak-dq current convention, Kv * Kt = 45 / pi. Catalog Kv conventions differ.
A rewind keeps copper volume fixed: Kt ~ turns, R and L ~ turns squared.
These are controlled model interventions, not manufacturer-qualified motors.
"""

from dataclasses import dataclass, asdict, replace
import math
import torch


@dataclass(frozen=True)
class Design:
    name: str
    axis: str = "baseline"
    kv_factor: float = 1.0
    peak_current_factor: float = 1.0
    gear_factor: float = 1.0
    voltage_factor: float = 1.0
    resistance_factor: float = 1.0
    cooling_resistance_factor: float = 1.0
    thermal_capacity_factor: float = 1.0
    added_mass_factor: float = 0.0
    added_rotor_inertia_factor: float = 0.0


def designs():
    return [
        Design("nominal"),
        Design("kv_low", "rewind Kv", kv_factor=0.67),
        Design("kv_high", "rewind Kv", kv_factor=1.5),
        Design("torque_low", "peak current", peak_current_factor=0.67),
        Design("torque_high", "peak current", peak_current_factor=1.5),
        Design("gear_low", "gear ratio", gear_factor=2 / 3),
        Design("gear_high", "gear ratio", gear_factor=4 / 3),
        Design("voltage_low", "bus voltage", voltage_factor=0.75),
        Design("voltage_high", "bus voltage", voltage_factor=1.25),
        Design("copper_low_R", "phase resistance", resistance_factor=0.5),
        Design("copper_high_R", "phase resistance", resistance_factor=2.0),
        Design("cooling_better", "thermal resistance", cooling_resistance_factor=0.5),
        Design("cooling_worse", "thermal resistance", cooling_resistance_factor=2.0),
        Design("mass_added", "actuator mass", added_mass_factor=1.0),
        Design("mass_added_double", "actuator mass", added_mass_factor=2.0),
        Design("rotor_heavier", "rotor inertia", added_rotor_inertia_factor=1.0),
        Design(
            "high_gear_with_mass",
            "combined design",
            gear_factor=4 / 3,
            added_mass_factor=1.0,
            added_rotor_inertia_factor=1.0,
        ),
        Design(
            "higher_current_with_cooling",
            "combined design",
            peak_current_factor=1.5,
            cooling_resistance_factor=0.5,
            thermal_capacity_factor=1.5,
            added_mass_factor=1.0,
        ),
    ]


def parameter_batch(base, cases, replicas, device):
    def values(field):
        return torch.tensor(
            [getattr(c, field) for c in cases], device=device
        ).repeat_interleave(replicas)[:, None]

    kv = values("kv_factor")
    peak = base.I_peak * values("peak_current_factor")
    return replace(
        base,
        lambda_m=base.lambda_m / kv,
        Rs=base.Rs * values("resistance_factor") / kv.square(),
        Ld=base.Ld / kv.square(),
        Lq=base.Lq / kv.square(),
        I_peak=peak,
        I_cont=torch.minimum(base.I_cont * kv, peak),
        gear_ratio=base.gear_ratio * values("gear_factor"),
        V_bus=base.V_bus * values("voltage_factor"),
        thermal_R_KperW=base.thermal_R_KperW * values("cooling_resistance_factor"),
        thermal_C_JperK=base.thermal_C_JperK * values("thermal_capacity_factor"),
    )


def describe(base, case, mass_unit, rotor_unit):
    kv = case.kv_factor
    kt = base.Kt / kv
    gear = base.gear_ratio * case.gear_factor
    peak = base.I_peak * case.peak_current_factor
    return {
        **asdict(case),
        "Kv_phase_peak_rpm_per_V": 45 / (math.pi * kt),
        "Kt_Nm_per_peak_q_A": kt,
        "gear_ratio": gear,
        "peak_current_A": peak,
        "continuous_current_floor_A": min(base.I_cont * kv, peak),
        "nominal_stall_peak_joint_Nm": kt * peak * gear * base.gear_eff,
        "base_speed_joint_rad_s_ideal_no_field_weakening": base.V_bus
        * case.voltage_factor
        / math.sqrt(3)
        / (base.pole_pairs * base.lambda_m / kv)
        / gear,
        "phase_R_ohm": base.Rs * case.resistance_factor / kv**2,
        "V_bus_V": base.V_bus * case.voltage_factor,
        "thermal_R_K_per_W": base.thermal_R_KperW * case.cooling_resistance_factor,
        "thermal_C_J_per_K": base.thermal_C_JperK * case.thermal_capacity_factor,
        "added_mass_per_actuator_kg": mass_unit * case.added_mass_factor,
        "added_motor_rotor_inertia_kg_m2": rotor_unit * case.added_rotor_inertia_factor,
        "added_joint_armature_kg_m2": rotor_unit
        * case.added_rotor_inertia_factor
        * gear**2,
    }


def add_carrier_mass(masses, inertias, carrier_ids, increments, radius):
    """Add compact spherical-equivalent mass at each parent link's existing COM.

    Shapes: mass [N,B], inertia [N,B,9], increments [N]. Multiple motors may
    share a carrier. This explicit placement assumption preserves existing COMs.
    """
    if (increments < 0).any() or radius <= 0:
        raise ValueError("Added mass must be nonnegative and radius positive")
    masses, inertias = masses.clone(), inertias.clone()
    for body in carrier_ids:
        masses[:, body] += increments
        for diagonal in (0, 4, 8):
            inertias[:, body, diagonal] += (2 / 5) * increments * radius**2
    return masses, inertias
