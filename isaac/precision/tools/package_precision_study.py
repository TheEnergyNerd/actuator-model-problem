"""Publish completed matched evaluations; never infer success from progress logs."""

import argparse, json, math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("results", type=Path)
p.add_argument("public", type=Path)
a = p.parse_args()
catalog = [
    ("baseline480_final", "Original policy · damping 0.1", 0.1),
    ("baseline_stable", "Original policy · damping 0.005", 0.005),
    ("hold_final", "Original policy + terminal hold · damping 0.005", 0.005),
    ("candidate_stable", "Fine-tuned policy · damping 0.005", 0.005),
]
rows = []
runs = []
for name, label, damping in catalog:
    path = a.results / (name + ".json")
    if not path.exists():
        continue
    j = json.loads(path.read_text())
    assert j.get(
        "evaluation_seconds", 60 if name == "baseline480_final" else None
    ) == 60 and [r["seed"] for r in j["results"]] == [201, 202, 203]
    stages = complete = drops = trials = 0
    for r in j["results"]:
        for flags, dwell, success, drop in zip(
            r["stage_success"], r["max_dwell_s"], r["complete"], r["drops"]
        ):
            assert flags == [v >= 0.5 - 1e-6 for v in dwell]
            assert success == (all(flags) and drop == 0)
            stages += sum(flags)
            complete += success
            drops += int(drop)
            trials += 1
    rows.append(
        dict(
            label=label,
            complete=complete,
            trials=trials,
            held_stages=stages,
            total_stages=6 * trials,
            drops=drops,
            mean_error_deg=sum(
                sum(r["mean_orientation_error_rad"]) for r in j["results"]
            )
            / trials
            * 180
            / math.pi,
        )
    )
    runs.append(dict(label=label, finger_damping_Nms_per_rad=damping, evaluation=j))
finished = len(rows) == 4
status = (
    "All four matched comparisons completed. The precision video uses the original policy with terminal hold; fine-tuned weights are not used in that recording."
    if finished
    else "Matched terminal-hold and fine-tuned-policy evaluations are still running. Only completed comparisons are shown."
)
criteria = "Six fixed rotation stages, including a physical push. Each needs 0.5 continuous seconds within 5.7°, angular speed below 0.5 rad/s, linear speed below 0.05 m/s, and position error below 0.12 m. All six holds and zero drops are required for a complete sequence. Each row uses 96 trials across seeds 201–203, with 480 Hz physics and 30 Hz policy actions."
(a.public / "hand/precision-study.json").write_text(
    json.dumps(dict(criteria=criteria, status=status, rows=rows), indent=2)
)
dest = a.public / "hand-precision"
dest.mkdir(parents=True, exist_ok=True)
(dest / "benchmark.json").write_text(
    json.dumps(dict(criteria=criteria, status=status, runs=runs), indent=2)
)
lines = [
    "# Allegro precision-control results",
    "",
    criteria,
    "",
    status,
    "",
    "| Policy/controller | Complete sequences | Stable stages | Drops | Mean error |",
    "|---|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['label']} | {r['complete']}/{r['trials']} | {r['held_stages']}/{r['total_stages']} | {r['drops']} | {r['mean_error_deg']:.1f}° |"
    )
lines += [
    "",
    "The full 60-second recording uses a separately fixed seed 42, not a selected successful evaluation environment. It completed 2/6 stages, had zero drops, and did not pass the complete sequence. Its longest continuous qualifying hold was 6.47 seconds. Video and measured 3D poses contain 1,800 synchronized frames.",
    "",
    "## Interpretation",
    "",
    "Changing damping alone did not improve overall stable-stage success in this comparison. The terminal controller and policy fine-tuning are separate interventions. No instantaneous orientation crossing counts as a stable hold. Thermal, torque, current, and voltage limits remain active. The experiment does not establish hardware performance or Rubik-solving ability.",
    "",
    "The original training target sampler uses X-then-Y rotations. The new curriculum expands arbitrary-axis rotations only after sustained holds. A short fine-tune therefore changes both the objective and target distribution; failure on harder rotations must be reported rather than hidden by a favorable clip.",
    "",
    "Raw outcomes include each trial’s stage flags, maximum continuous dwell, drops, orientation error, torque shortfall, winding temperature, and checkpoint hash. Development pilots are retained in the source repository separately from these matched comparisons.",
]
if finished:
    lines += ["", "## Decision", "", "Terminal hold increased stable stages from 87/576 to 227/576 under matched controller settings, but complete sequences remained 0/96. Mean orientation error increased from 33.1° to 37.8° and drops from 4 to 5. This is an improvement in individual holds, not overall reliable dexterity.", "", "The fine-tuned candidate achieved 3/576 stable stages, 0/96 sequences, and 19 drops. It was rejected and is not used by the demo. The precision curriculum and runner are retained as experimental code, with this failure documented."]
(dest / "benchmark.md").write_text("\n".join(lines) + "\n")
print(json.dumps(rows, indent=2))
