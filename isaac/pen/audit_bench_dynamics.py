"""Cross-check the fast actuator approximation against substepped dq dynamics.

This checks software consistency, not agreement with hardware. Includes reversals
at declared constant speeds; the external speed source is a virtual dynamometer.
"""

import json
from pathlib import Path
import sys
import numpy as np
from bench_profile import electrical_parameters, KT

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / "precision/atlas_actuators_ext/atlas_actuators"
    ),
)
from foc_core import FOCActuatorParams, step_current_lag, step_dq


def audit():
    rpm = np.array([0, 1000, 2000, 3000, 3400], dtype=float)
    p = FOCActuatorParams(
        **electrical_parameters(),
        V_bus=12,
        I_peak=2,
        I_cont=1,
        gear_ratio=1,
        gear_eff=1,
        thermal_R_KperW=10,
        thermal_C_JperK=8,
        T_ambient_C=25,
        T_derate_C=85,
        alpha_pm_perC=0
    )
    lag = [np.zeros_like(rpm), np.zeros_like(rpm), np.full_like(rpm, 25)]
    dq = [np.zeros_like(rpm) for _ in range(4)] + [np.full_like(rpm, 25)]
    trace = []
    for i in range(240):
        requested = np.full_like(rpm, KT * (1.5 if i < 120 else -1.5))
        fast, *lag = step_current_lag(p, requested, rpm * 2 * np.pi / 60, *lag, 1 / 240)
        exact, *dq = step_dq(
            p, requested, rpm * 2 * np.pi / 60, *dq, 1 / 240, substeps=100
        )
        assert np.isfinite(exact).all()
        trace.append(
            dict(
                t=(i + 1) / 240,
                requested=requested.tolist(),
                lag_torque=fast.tolist(),
                dq_torque=exact.tolist(),
            )
        )
    fast = np.array([r["lag_torque"] for r in trace])
    exact = np.array([r["dq_torque"] for r in trace])
    settled = np.r_[np.arange(60, 120), np.arange(180, 240)]
    return dict(
        status="Numerical cross-check; not hardware validation",
        rpm=rpm.tolist(),
        substeps=100,
        electrical_integration_Hz=24000,
        physics_Hz=240,
        max_transient_difference_Nm=float(abs(fast - exact).max()),
        max_settled_difference_Nm=float(abs(fast[settled] - exact[settled]).max()),
        trace=trace,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = audit()
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}, indent=2))
