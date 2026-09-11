# Rubik / Wuji results — 2026-09-11

The browser prototype uses the native five-finger Wuji Hand 2, a 26-cubie
PhysX mechanism, a recorded Isaac video and the same measured poses in 3D.
**A stable full scramble solve and a free bimanual face turn are not validated.**

## Recorded prototype

`sequence21`: one `U` quarter-turn from `U'`, fixed core, 651 measured frames at
50 FPS, 1 kHz physics, 53 bodies. **Passed a continuous 0.4-second solved hold
after release**, independently recomputed from `poses.bin`. Final sticker error
is 0.3476°; maximum anchor separation over the entire run is 0.1880 mm, below the
0.5 mm threshold. The first grasp attempt passed in 13 seconds.

It uses explicit rigid high-friction fingertip contacts, 6 mm commanded grip
closure, 0.0002 Nms/rad passive damping, and up to 45° wrist overtravel to
compensate for slip. Wrist travel and actual cube travel are distinct. There
are no cube motors or cube pose writes during the rollout. This is a fixture
solve of a one-turn scramble, not a free hand or complete scramble solve.

The video is an Isaac render of the same measured states, exported after the
physics run. Rendering disables physics and sets the recorded body poses;
it does not run a second simulation or invent successful motion. `render.json`
records this distinction. The browser replay uses those same poses and native
Wuji geometry.

Earlier `sequence02` ended in a solved sticker state, but failed the new
continuous hold criterion. Its original final-frame result is retained in
`results/sequence02.json`; it is superseded by sequence21 in the browser.

## Contact and model findings

- Ordinary USD traversal found zero of the hand's collision meshes because
  they were instances. Explicitly deinstancing them exposes all 21 colliders;
  configuration now fails if that count is wrong.
- Restored ten native wrist/finger-base collision exclusions per hand.
  Both hands pass a free-space grasp tracking check: maximum joint error
  below 8e-9 rad and maximum actuator torque below 7e-8 Nm. This verifies model
  assembly and joint control; it does not verify grasping.
- Enabled contact processing so per-cube contact force measurements work.
- Initialize angular joint state before physics starts, and reset both root
  and joint velocities. Camera warmup must not change the starting state.
- Stock, rigid high-friction, and compliant pad profiles are explicit in each
  new run's configuration. Pad parameters are experiments, not measured Wuji
  hardware properties.
- Sequence control releases, retracts and allows settling before validation.
  Failed turns can regrasp from a different direction without teleporting any
  body. Optional wrist overtravel stops when the measured cube goal is held.

## Selected experiments

| Run | Scope | Observed outcome |
| --- | --- | --- |
| sequence20 | One-turn fixture solve, rigid contacts and stronger damping | Passed after three grasp attempts, 49.84 s. |
| sequence21 | Same contact model with bounded wrist overtravel | Passed on the first attempt, 13.00 s. |
| sequence22 | Full reference scramble, same controller | Six of twelve quarter-turns passed strict release/hold validation; D' failed after four attempts. 134.26 s recorded. |
| sequence15 | Reference scramble inverse, fixed core | Reached five quarter-turn checkpoints, then failed B'. Checkpoints used the older final-frame criterion. |
| sequence18 | Fixed core, stock contacts, release and regrasp recovery | Completed two quarter turns; failed U after four attempts. |
| bimanual17 | Reference starting grasp, native joint initialization | No useful left cube contact; dropped at 1.72 s. |
| bimanual18 | Deeper support pinch | Held initially, then slipped under turning load; dropped at 5.60 s. |
| bimanual19 | Gradual grip closure and third support finger | Self-contact loads and loss of cube support; dropped at 6.00 s. |

The full sequence22 recording is also exposed in the browser, including all
four failed attempts at its seventh turn. Its six passing checkpoints occur
at 13.00, 24.48, 35.74, 47.08, 58.42 and 71.40 seconds. The seventh turn remains
incorrect at 134.26 seconds. Its peak anchor separation reaches 0.6175 mm during
failed manipulation; this does not meet the 0.5 mm connection threshold.

Raw configurations/results and independent pose audits are retained in `results/`. These trials changed
multiple experimental parameters and are not controlled actuator comparisons.
No success rate or hardware advantage is inferred from them.

## Limits and next work

The passive cube approximates a Rubik mechanism with spherical/revolute
joints, geometric detents and axis guides. It has not been calibrated against
a real cube's torque or compliance. It does not reproduce the reference's
joint armature or radial compliance. The wrists are bounded Cartesian force
servos, not complete robot arms. Grasp IK ignores collision constraints;
adding a fingertip target can introduce self-collision, as bimanual19 shows.

The next substantive requirement is a collision-aware supporting grasp and
turn control that generalizes across all faces in a full sequence. Existing failed trials must remain marked failed.
