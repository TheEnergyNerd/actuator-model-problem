"""Report paired per-seed sequence outcomes without selecting attractive videos."""

import argparse, json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("evaluations", type=Path, nargs="+")
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
rows = []
for path in a.evaluations:
    j = json.loads(path.read_text())
    seeds = []
    for r in j["results"]:
        stage = np.asarray(r["stage_success"])
        dwell = np.asarray(r["max_dwell_s"])
        complete = np.asarray(r["complete"])
        drops = np.asarray(r["drops"])
        assert np.array_equal(stage, dwell >= 0.5 - 1e-6)
        assert np.array_equal(complete, stage.all(-1) & (drops == 0))
        seeds.append(
            dict(
                seed=r["seed"],
                trials=len(complete),
                complete_sequences=int(complete.sum()),
                stage_success_counts=stage.sum(0).tolist(),
                drop_events=int(drops.sum()),
                mean_orientation_error_deg=float(
                    np.mean(r["mean_orientation_error_rad"]) * 180 / np.pi
                ),
                mean_torque_shortfall_fraction=float(
                    np.mean(r["torque_shortfall_fraction"])
                ),
                max_winding_C=max(r["peak_winding_C"]),
            )
        )
    rows.append(
        dict(
            source=str(path),
            checkpoint_sha256=j["checkpoint_sha256"],
            seeds=seeds,
            total_complete_sequences=sum(s["complete_sequences"] for s in seeds),
            total_trials=sum(s["trials"] for s in seeds),
        )
    )
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
