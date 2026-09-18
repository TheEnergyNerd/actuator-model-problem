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
