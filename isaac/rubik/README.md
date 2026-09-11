# Wuji / Rubik in Isaac Lab

This is a contact-driven Rubik manipulation experiment using **Wuji Hand 2**,
with its native, MIT-licensed Isaac USD geometry. It does not run MuJoCo or use
reference-demo motion as a replay. The published prototype fixes the cube core;
a stable complete solve and a free two-hand face turn have not been validated.
See [RESULTS.md](RESULTS.md) for the measured outcomes and failed trials.

## Mechanism and controller

- 26 individually simulated cubelets: six revolute centers, twenty spherical
  edge/corner joints around a common core. Pitch 19.05 mm, total width 56.6 mm.
- No cube motors. Passive cubic-symmetry detents and an axis guide depend only
  on current relative orientations. Equal/opposite torque is applied to the core.
- Native Wuji finger collision meshes, joint limits, and mass/inertia data.
  Finger position control uses 3 Nm/rad stiffness, 0.08 Nms/rad damping and
  a 1.5 Nm **experimental simulation limit**, not a verified hardware rating.
- A floating wrist is driven by bounded external force/torque (30 N per axis,
  2 Nm per axis). This is a Cartesian test mount, not a simulated robot arm.
- URDF forward kinematics and bounded fingertip IK determine pinch targets.
- Cube pose writes occur only during initialization. Playback uses recorded
  body states, never the logical cube solver or wrist trajectory.
- Facelets are decoded from actual cubie orientations. Alignment alone is not
  success: the decoded facelets must match the expected move. The numerical
  worst-body rotation includes center orientation, which can exceed visible
  sticker error. New runs require a 0.4-second matching hold and less than
  0.5 mm anchor separation after release.

`sequence_control.py` supplies approach, pinch, turn, verification, release, retract and
settling commands for a move sequence. Its existence is not evidence of a
successful physical solve. `cube_state.py` implements move algebra and a strict
geometric facelet decoder. It does not establish that arbitrary measured
facelet permutations are reachable; matching an expected legal sequence does.

## Run

Use Isaac Lab 2.1 / Isaac Sim 4.5 on the existing RunPod image:

```sh
./isaaclab.sh -p /workspace/atlas-rubik/probe.py --headless \
  --device cpu --scramble "U'" --moves U --feedback \
  --closure .006 --contact-profile rigid --joint-damping .0002 \
  --overtravel-deg 45 --output /workspace/atlas-rubik/new-run
```

Outputs: video (when requested), raw states (`poses.npz`), 50 FPS float32 replay
positions/quaternions (`poses.bin`, wxyz), mesh geometry, telemetry, configuration,
and a measured pass/fail result. Output directories must not already exist.
Do not compare CPU/GPU runs as identical recordings; they use different PhysX
execution paths. Compare the video and replay exported from **one** run.

Local dependency-light checks:

```sh
python -m unittest discover -s isaac/rubik -p 'test_*.py'
```

Requires NumPy/SciPy, PyTorch and usd-core. The USD builder additionally requires the `pxr` modules
provided by Isaac Sim (local model inspection can use `usd-core`).

The browser offers the validated single turn and the complete recorded full
scramble attempt, with its failures preserved. To audit poses independently:

```sh
python isaac/rubik/audit_recording.py lab-src/public/data/rubik
python isaac/rubik/audit_recording.py lab-src/public/data/rubik/full-attempt
```

## Contact profiles and bimanual experiment

`probe.py --contact-profile stock|rigid|pad` selects explicit contact settings.
`rigid` uses high friction and a finite torsional patch; `pad` additionally uses
experimental 5000 N/m compliant distal contact. Results record the actual
values. No profile is a validated hardware specification.

`bimanual.py` prepares a two-hand grasp on a temporary fixture and disables
that fixture before the turn. It currently fails. `--third-support` is an
experimental extra finger target and is not collision-aware. Do not promote
its recordings as a successful bimanual demonstration.

`render_recording.py --recording /path/to/run` renders measured `poses.npz`
against that run's `scene.usda` with physics disabled during rendering. It does
not invent intermediate states. `render.json` distinguishes this export from
live simulation video. A 50-frame recorded-state smoke test produced the
expected 25-frame / one-second video. Add `--video` to the physics runner for
live capture instead. Keep video and replay from the same run.

## Provenance

- Wuji native USD/URDF: https://github.com/wuji-technology/wuji-description
  revision `4c1073d0a3ad1daaf6546d219db751d8448d3888`.
  License and file SHA256 manifest are in `assets/wuji/`.
  `fetch_wuji.py` re-fetches these exact files and verifies their digests.
- Reference demonstration: https://dex-rubik-cube.yanjieze.com/reproduce.html
  Its public numerical grasp seeds inform the initial pinch IK, and its
  mechanism description informs the passive cube design. Our simulator,
  controller, cube builder, decoder and recorder are separate implementations.
  Its source, videos and replay are not redistributed here.
- Reference fixed scramble: `R U F' L2 D B R' U2 F D'`.
  Its inverse is `D F' U2 R B' D' L2 F U' R'`.
  Verifying this algebra is not a physical manipulation result.
