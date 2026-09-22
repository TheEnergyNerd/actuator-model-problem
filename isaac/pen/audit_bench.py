"""Validate the supplied electrical calibration without contacting hardware.

Usage: python audit_bench.py /path/to/moteus-cal-....log --out evidence.json
Output excludes device serial/UUID. Raw file identity is preserved by SHA256.
"""

import argparse
import hashlib
import json
from pathlib import Path
from bench_profile import MEASURED, FIRMWARE, RAW_LOG_SHA256, provenance


def audit(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RAW_LOG_SHA256:
        raise ValueError("This profile requires the reviewed raw calibration log")
    data = json.loads(raw)
    mapping = {
        "winding_resistance": "Rs",
        "inductance_d": "Ld",
        "inductance_q": "Lq",
        "kv": "Kv_line_line_peak_rpm_per_V",
    }
    for source, target in mapping.items():
        if data[source] != MEASURED[target]:
            raise ValueError(f"Calibration mismatch: {source}")
    if data["calibration"]["poles"] != 2 * MEASURED["pole_pairs"]:
        raise ValueError("Pole count mismatch")
    if data["device_info"]["git_hash"] != FIRMWARE or data["py_version"] != "1.1.1":
        raise ValueError("Source convention audit does not cover this firmware/tool")
    result = provenance()
    result.update(
        raw_values_verified=True,
        scalar_inductance_H=data["inductance"],
        torque_bandwidth_requested_Hz=data["torque_bw_hz"],
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(json.dumps(audit(a.log), indent=2))
