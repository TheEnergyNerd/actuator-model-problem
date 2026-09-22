# Sharpa pen reproduction and physical sensitivity tests

Isaac Lab 2.3.2 / Isaac Sim 5.1, 240 Hz physics, 60 Hz control, the pinned
Dexterous Astra checkpoint (iteration 1728). Atlas did not train this checkpoint.
32 development seeds, 41000000–41000031. Prepared grasps; no pickup task.

| Configuration | Three timed turns + supported 1 s hold | Additional native contact/finger checks | Trial 0 third turn |
|---|---:|---:|---:|
| Native Sharpa | 31/32 | 5/32 | 5.15 s |
| 25% joint torque ceilings | 28/32 | 2/32 | 5.15 s |
| Hand-body mass and inertia ×1.5 | 32/32 | 7/32 | 5.50 s |

Trial 0 was selected before evaluation in each condition. It completes the motion
task in all three conditions but fails the separate thumb-release requirement.
All numerical outcomes, including failures, are published.

The interventions apply after grasp preparation. Initial joint angles, pen poses,
pen mass and friction are bitwise identical across conditions for every seed.
The torque limits are verified by reading back PhysX DOF maximum forces; changed
body masses are also read back. Mass and inertia scale together, including the
fixed palm body; the wrist remains fixed. These are sensitivity tests, not
manufacturable motor designs, Kv sweeps or thermal-model evaluations. There is
no retraining. The extra-mass result does not establish a better design.

The motion gate includes turn timing, supported hold, no drop before completion,
joint limits and valid initial settling. The additional gate includes finger
participation, thumb release and penetration. Native PhysX contact separations
replace the upstream independent MuJoCo hull audit; initial penetration was not
recorded by the reference. Therefore the stricter result is not a reproduction
of the upstream acceptance score or an estimate of hardware reliability.

Video and 3D replay use the same recorded rigid-body states. Neither motion nor
contact is reconstructed by a kinematic controller. Studio lighting, floor,
colors and mesh-coordinate rounding to 0.1 micrometre are display-only changes.
The reported joint drive torque is the implicit actuator's estimate, not a
physical torque sensor measurement. No current or winding temperature is inferred.

Raw evaluation trajectories and the checkpoint hash are retained under `results/`.
The public replay is `lab/data/pen/`; it includes all per-trial outcomes and the
matched initial-state comparison. Bootstrap failures before physics evaluation
are retained in the local run workspace and are not counted as task trials.

## Matched explicit PD comparison — September 18

| Model | Motion | Additional contact gate | Trial 0 third turn |
|---|---:|---:|---:|
| Fixed torque ceiling | 26/32 | 3/32 | 8.93 s (fails timing/hold) |
| Cold FOC/current-lag model | 27/32 | 2/32 | 5.35 s |
| Same motor model, 100°C start | 25/32 | 4/32 | 5.13 s |

All three use identical explicit PD, gains, native nominal stall ceilings, mass,
inertia and checkpoint. Initial q0, pen poses, mass and friction match bitwise for
all seeds. The earlier native implicit runs are a separate controller reference.

Assumptions: 12 V bus, 2 A peak/1 A continuous, 2 Ω phase resistance, 0.012 Nm/peak-q-A,
200 µH Ld/Lq, 7 pole pairs, 80% gear efficiency, joint gearing chosen to match the
native cold stall ceilings. Thermal R = 10 K/W, C = 8 J/K, linear current derating
from 2 A at 25°C to 1 A at 85°C. Magnet temperature dependence is disabled because
only winding temperature is modeled. Phase-peak Kv is 1194 RPM/V; conventions matter.

Cold peak winding temperature was 30.073°C and peak dq current 1.925 A. The largest
recorded requested/delivered torque difference was 8.05e-7 Nm: constraints did not
bind. Contact-sensitive trajectory divergence from tiny numerical differences
must not be presented as a cold motor-design penalty or benefit. The hot-start
model cooled to approximately 89.6°C and clipped up to 0.932 Nm. No strong ranking
is established by these 32 development seeds. All three pass trial 1's motion gate.

All motor trace values were finite. Current remained within the temperature-dependent
circle (floating-point tolerance); the steady-state dq voltage check stayed below
12/sqrt(3) V on recorded samples, peaking at 4.79 V cold / 3.75 V hot. This sample
audit does not validate switching transients or hardware fidelity.

Both trial 0 and trial 1 have video and measured body-state replay. Trial 1 was
chosen as the first simplified success before inspecting motor outcomes, not as
a selected motor failure. Default comparison is cold versus fixed torque; hot
is available in the selector. Parameter and selection provenance accompany each replay.

## Equal-budget policy fine-tuning — September 18

524,288 training transitions per condition, a common external checkpoint, separate
training seeds 43000000–43000127 and test seeds 42000000–42000063.
The final iteration-64 checkpoint was fixed before evaluation. One training seed.

| Policy | Fixed torque | Cold motor | 100°C motor |
|---|---:|---:|---:|
| Original frozen policy | 48/64 | 47/64 | 47/64 |
| Simplified-model fine-tuning | 56/64 | 56/64 | 56/64 |
| Motor-model fine-tuning | 47/64 | 50/64 | 52/64 |

The motor-trained policy completed 4 fewer hot-start trials than the simplified-trained policy in this batch; more detailed training dynamics did not automatically improve this skill. This pilot does not establish a benefit across training seeds.

All nine conditions have bitwise-identical initial joint angles, pen poses, pen
mass and friction for each test seed. Actor weights changed in both fine-tunes;
observation normalizers remained identical to the reference checkpoint. Training
reward is an Atlas surrogate, not the unavailable upstream training implementation.
The stricter contact gate remains separately reported. No hardware calibration or
training from scratch is claimed. See TRAINING.md and the public training manifest.

Raw trajectories, motor traces and final checkpoints are retained in the local
`/root/atlas/sharpa-training-run/` workspace. Public artifacts include every test
outcome, trial-0 measured poses/telemetry/video, learning logs, parameter provenance
and checkpoint/raw-data hashes. Rendered videos are 720×720, 60 FPS; physics is 240 Hz.

## Bench-informed virtual actuator — three training seeds

Cold motor test, motion success: simplified-trained mean 95.6% (seed range 92.2–97.7%); motor-trained mean 95.1% (range 92.2–96.9%).
100°C-start stress test, motion success: simplified-trained mean 95.8% (seed range 94.5–97.7%); motor-trained mean 94.5% (range 91.4–96.1%).
Three training seeds provide replication; the shared test simulator and assumed transmission/cooling do not establish real-hand performance.

| Training seed | Test plant | Simplified trained | Motor trained |
|---|---|---:|---:|
| 44000000 | ideal | 122/128 | 127/128 |
| 44000000 | motor | 118/128 | 124/128 |
| 44000000 | motor-hot | 121/128 | 123/128 |
| 44010000 | ideal | 122/128 | 119/128 |
| 44010000 | motor | 125/128 | 123/128 |
| 44010000 | motor-hot | 122/128 | 123/128 |
| 44020000 | ideal | 123/128 | 119/128 |
| 44020000 | motor | 124/128 | 118/128 |
| 44020000 | motor-hot | 125/128 | 117/128 |

motor: torque delivery differs from the clipped request by more than 0.001 Nm in 0.030% of recorded joint samples, averaged across six policies. Largest difference: 0.008 Nm. This includes current-loop lag and limits; samples are recorded at 60 Hz.

motor-hot: torque delivery differs from the clipped request by more than 0.001 Nm in 0.268% of recorded joint samples, averaged across six policies. Largest difference: 1.093 Nm. This includes current-loop lag and limits; samples are recorded at 60 Hz.

Validation motion success (%), mean across three seeds:
[
  {
    "iteration": 64,
    "ideal": 85.41666666666666,
    "motor": 82.29166666666666
  },
  {
    "iteration": 128,
    "ideal": 92.70833333333334,
    "motor": 89.58333333333334
  },
  {
    "iteration": 256,
    "ideal": 91.66666666666666,
    "motor": 90.625
  }
]

All six policies received 4,194,304 transitions. Fixed final checkpoint at iteration 256; cold training starts; 18 independent validation evaluations and 21 final evaluations. Full per-trial motion and additional-contact outcomes are published in `lab/data/pen/bench/`. The replay uses predeclared training seed 44000000 and test seed 46000000.

Electrical calibration was checked against the supplied raw log and source conventions. Torque constant is derived, not measured. See BENCH_STUDY.md for virtual remote-drive, gearing and thermal assumptions. This study does not demonstrate hardware transfer.

Training checkpoints and raw evaluation arrays are retained in the local study workspace; public manifests contain checkpoint and raw-array hashes. Prior raw NPZ results remain available in the GitHub repository.
