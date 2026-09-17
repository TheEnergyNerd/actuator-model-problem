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
