"""Electrical calibration provenance for a virtual remote-drive design.

The measured motor is mj5208, NOT the Sharpa hand motor. Gearing, transmission,
current limits, cooling and packaging are engineering assumptions. No hardware
transfer or winding-temperature accuracy is established by this profile.
"""

import math

RAW_LOG_SHA256 = "1e1b08b78c474780261213c34723f02d504bbc77c3140a81100e5b2ac133b4bc"
FIRMWARE = "6f063a90a97012dc9f6ccc420390c5a6cc55fac7"
MEASURED = dict(
    Rs=0.06507539791011802,
    Ld=2.7011447531700632e-5,
    Lq=4.524194701244842e-5,
    pole_pairs=7,
    Kv_line_line_peak_rpm_per_V=310.6750471286357,
)
# moteus 1.1.1 Kv is RPM / peak line-line voltage. Atlas uses phase-peak
# back EMF and peak dq current. This agrees with the pinned firmware's
# kTorqueFactor = (3/2) * (1/sqrt(3)) * (60/(2*pi)). Not a load-cell fit.
PHASE_KV = math.sqrt(3) * MEASURED["Kv_line_line_peak_rpm_per_V"]
KT = 45 / (math.pi * PHASE_KV)


def electrical_parameters():
    return dict(
        Rs=MEASURED["Rs"],
        Ld=MEASURED["Ld"],
        Lq=MEASURED["Lq"],
        pole_pairs=7,
        lambda_m=KT / (1.5 * 7),
        current_bw_hz=200,
    )


def provenance():
    return dict(
        profile="mj5208-virtual",
        status="Controller-calibrated electrical parameters; virtual remote-drive transmission; unvalidated thermal model",
        raw_log_sha256=RAW_LOG_SHA256,
        firmware_revision=FIRMWARE,
        tool_version="1.1.1",
        measured=MEASURED,
        derived=dict(Kt_Nm_per_peak_q_A=KT, phase_peak_Kv_rpm_per_V=PHASE_KV),
        assumptions=[
            "12 V bus, 2 A peak and 1 A continuous drive limits",
            "Per-joint gearing preserves the native nominal cold stall ceiling, efficiency 0.8",
            "Virtual remote motors: native hand link mass/inertia retained; no packaging claim",
            "Thermal R=10 K/W, C=8 J/K, ambient/reference=25 C, derate endpoint=85 C",
            "200 Hz requested calibration bandwidth approximated as first-order current lag, not measured bandwidth",
            "No measured backlash, cable compliance, friction, rotor inertia or thermal validation",
        ],
        sources=[
            f"https://github.com/mjbots/moteus/blob/{FIRMWARE}/fw/bldc_servo.cc",
            "https://pypi.org/project/moteus/1.1.1/",
        ],
    )
