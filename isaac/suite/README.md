# Atlas demonstration suite

The agreed scope includes all three tracks below, plus actuator design optimization across them. All simulation remains in Isaac Lab / PhysX. Existing recordings remain available; new demonstrations require measured performance before publication.

| Track | First milestone | Later demonstrations | Acceptance evidence |
| --- | --- | --- | --- |
| Robot Olympics | ANYmal rough-terrain traversal | Stairs, gaps, payload carry, disturbance recovery; humanoid extension after a reliable walking baseline | Course completion, falls, speed, foot slip, energy, winding temperature |
| Two-hand workshop | Stable free-object grasp and controlled rotation | Regrasp, cap opening, tool pickup and fastening; continue the Wuji Rubik task | Object pose, sustained contact, completed operation, drops, recovery trials |
| Precision assembly | Factory peg insertion | Gear meshing and nut threading, followed by randomized tolerances | Insertion depth, alignment, completion time, contact force, jam and damage limits |
| Actuator optimization | Matched evaluations of real actuator variants | Search Kv, gearing, current limits, motor mass and thermal design | Held-out success and a Pareto comparison of energy, time, mass and heat |

## Current evidence

Flat-ground ANYmal/G1 and Allegro recordings already exist. Wuji supports an independently checked unsupported stationary hold; one free-cube quarter-turn now passes independent pose validation, while multi-face sequences fail. The fixed-core quarter-turn recording is a separate fixture experiment. None of these results establish parkour, tool use or assembly performance.

`smoke.py` checks native task creation, reset and finite stepping on the installed Isaac Lab version. Zero actions are deliberately labelled as an infrastructure test. A passing smoke test does not validate a policy, obstacle course or assembly operation. It records all early terminations and never assigns task success.

Run each environment in a fresh Isaac process, with a fresh output directory:

```bash
"$ISAACLAB/isaaclab.sh" -p isaac/suite/smoke.py --headless --device cuda:0 \
  --task Isaac-Factory-PegInsert-Direct-v0 --output /path/to/new-peg-smoke
```

Other native starting points: `Isaac-Factory-GearMesh-Direct-v0`, `Isaac-Factory-NutThread-Direct-v0`, and `Isaac-Velocity-Rough-Anymal-C-Play-v0`. Keep their native physics settings for these checks. Use an external process timeout when running unattended; absence of a result is a failure to finish, not a passing check.

The batch runner enforces a six-minute timeout per task and writes a log, exact command and aggregate result. It launches sequentially to limit interference with existing training:

```bash
python3 isaac/suite/run_smokes.py --isaaclab "$ISAACLAB" \
  --extension isaac/precision/atlas_actuators_ext --output /path/to/new-suite-check
python3 -m unittest discover -s isaac/suite -p 'test_*.py'
```

The new `Isaac-Velocity-Rough-Anymal-C-AtlasFOC-v0` registration uses the native rough-terrain curriculum and RSL-RL training configuration with Atlas FOC legs. It is a starting point for training, with unchanged robot mass; it has no associated trained rough-terrain checkpoint yet.

## RunPod results, 2026-09-11

All five environment checks passed: native rough ANYmal, Atlas rough ANYmal, peg insertion, gear meshing and nut threading. Each ran two environments for 60 control steps at native timestep/decimation, with finite observations/rewards and no early termination. Raw summaries are in `results/native-01`. These are startup checks, not competent-task results.

An official NVIDIA RSL-RL rough-terrain checkpoint was available at:

https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Rough-Anymal-C-v0/checkpoint.pt

SHA256: `7efbaef126790c3f985d80b674712678636fe7c9a3a09ad887a69f5674f7adaf`.

The stock-policy pilot ran eight environments for 20 seconds each on seed 201, with zero failure terminations and one normal episode timeout per environment. Per-environment mean planar velocity tracking error ranged from 0.078 to 0.169 m/s. This is a short development baseline on the native terrain distribution, not a held-out course evaluation. Reset transitions are retained in the exported state and excluded from velocity-error aggregation.

The fixed-policy Atlas transfer pilot also completed. It recorded one failure termination, versus zero for stock, and mean per-environment planar tracking error of 0.167 m/s versus 0.118 m/s. The Atlas run changes the native learned ANYdrive actuator model to explicit FOC dynamics with PD gains; this is not an isolated Kv or gearing intervention. Native heading feedback produces different yaw commands as trajectories diverge. Planar commands first diverged at 5.86 seconds after the Atlas failure/reset. Do not treat the aggregate as a fully paired causal hardware estimate.

Both pilots retain all 1,000 measured frames for eight robots (17 bodies each) in `results/anymal-stock` and `results/anymal-atlas`, with telemetry, result metadata and SHA256 manifests. Pose shape, finiteness, quaternion norms, 50 Hz timestamps and termination counts were independently checked. The current evaluator additionally exports native terrain meshes, body poses and synchronized Isaac renders with `--record`; the browser includes environment 0 of the stock pilot.

```bash
"$ISAACLAB/isaaclab.sh" -p isaac/suite/eval_anymal.py --headless --device cuda:0 \
  --checkpoint /path/to/checkpoint.pt --output /path/to/new-baseline
# Fixed-policy Atlas comparison: same native PLAY terrain/events/commands.
PYTHONPATH=isaac/precision/atlas_actuators_ext "$ISAACLAB/isaaclab.sh" \
  -p isaac/suite/eval_anymal.py --headless --device cuda:0 --atlas \
  --checkpoint /path/to/checkpoint.pt --output /path/to/new-atlas-baseline
```

The native Factory checkpoint URLs constructed by the installed Isaac Lab helper returned HTTP 404 for all three assembly tasks. This does not prove no compatible checkpoint exists elsewhere; we trained a native peg-insertion policy locally on RunPod. A development checkpoint passed sustained insertion in 42/48 trials (seed 201, 16 environments, three episodes). See `results/peg-pilot/result.json`; this retains native prepared grasps and does not establish gear-meshing or nut-threading competence.

## Shared experiment protocol

1. Establish a competent controller with the stock robot and record checkpoint/configuration hashes. Workshop tasks require their own collision-aware controller and mechanics validation.
2. Introduce the Atlas actuator model and verify delivered torque, voltage/current saturation, friction and thermal integration. A changed motor mass must change the physical link mass and inertia with a documented mounting transform; changing a UI field is not a mass experiment. Keep Kv and Kt physically consistent under the chosen motor convention.
3. Evaluate a fixed policy across actuator variants to measure robustness. Separately retrain per variant to measure achievable performance. Never mix these comparisons.
4. Use identical task distributions and held-out seeds across variants. Report every trial, success rate, uncertainty and failure modes. Development videos cannot replace evaluation.
5. Export video and body poses from the same physical run, synchronized by simulation timestamp. Mark resets, interventions and fixture constraints. Reuse the existing orbitable replay and side-by-side comparison layout.
6. Optimize only after this evaluation is reliable. Keep real mass/inertia and thermal parameters; artificial accelerated heating is a separately labelled experiment.

The first progression is native environment validation → stock-policy baseline → actuator integration → held-out evaluation → published video/replay. This applies to each track independently so a polished interface cannot conceal an unfinished controller.

## Complete course recordings

`lab-src/public/data/course/manifest.json` lists all six matched development runs. Native ANYdrive and Atlas nominal passed. Higher Kv, lower gearing and +6 kg added carrier mass reached the finish but left the obstacle corridor. Reduced current produced a failure termination. The corridor criterion was added during post-run auditing, not specified before these development runs. The original finish flag remains in `source-result.json`; `audit.json` records the stricter result, geometry checks and artifact hashes.

```bash
PYTHONPATH=isaac/precision/atlas_actuators_ext "$ISAACLAB/isaaclab.sh" -p isaac/suite/eval_anymal.py --headless --device cuda:0 --course --record --num-envs 1 --atlas --design nominal --checkpoint /path/to/checkpoint.pt --output /path/to/new-course
"$ISAACLAB/isaaclab.sh" -p isaac/suite/render_recording.py --headless --device cpu --recording /path/to/new-course
python3 isaac/suite/audit_course.py lab-src/public/data/course/nominal
```

The winding variant changes Kt, resistance and inductance consistently with Kv. Gear changes affect the motor model; native armature is retained. The mass experiment adds 0.5 kg per actuator at its carrier COM with spherical-equivalent inertia (0.04 m radius), and checks PhysX mass readback. These are controlled simulation interventions, not manufacturer-qualified designs.

For a local preview use `python3 tools/serve_lab.py`. Its HTTP byte-range support is necessary for reliable video seeking.

## Assembly recordings and queue

The first recorded peg episode uses the same evaluated checkpoint (SHA256 `3e3600aa8f01ea9c551cfcec1c99b72aae545400bfff20c66de02e13869820e8`). It was chosen before evaluation: environment 0, episode 0. The recorder captures robot, fixture and peg bodies in the reward hook, before the native terminal reset. The complete episode is retained at the native 15 Hz control rate; physics runs at 120 Hz. The repeat evaluation reproduced 42/48 successes.

`train_assembly.py` runs gear meshing followed by nut threading after this session's peg trainer finishes. Each training process has a two-hour limit and a 200-iteration cap, followed by a separate deterministic evaluation and first-episode recording. A timeout is recorded explicitly; reaching the cap does not itself establish competence. Checkpoints remain on RunPod and are copied with ZIP integrity and concurrent-write checks before evaluation. This queue is running at `/workspace/atlas-suite/assembly-queue-20260911`; it has not completed. The existing pod shutdown watcher retains its deadline and waits for the additional queue process before treating the original hand job's completion as a reason to stop.
