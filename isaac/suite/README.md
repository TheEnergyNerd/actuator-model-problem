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

## Downloadable candidate designs

The course viewer now shows nominal/selected parameters, calculated unloaded joint and motor RPM, mechanical effects and an editable model-candidate builder. Its exported JSON is accepted by `eval_anymal.py --atlas --course --design-file candidate.json`. Three untested examples are in `candidates/`: torque-focused, speed-focused, and higher-current with added carrier mass. These are simulation parameter sets, not manufactured motor designs. Their outcome is not known and no existing video is relabelled as their result.

```
PYTHONPATH=isaac/precision/atlas_actuators_ext "$ISAACLAB/isaaclab.sh" -p isaac/suite/eval_anymal.py --headless --device cuda:0 --atlas --course --record --num-envs 1 --checkpoint /path/to/checkpoint.pt --design-file isaac/suite/candidates/torque_focused.json --output /path/to/new-candidate-run
```

The parser rejects non-finite or unsupported parameters before launching Isaac. Current changes use the existing FOC parameter batch, winding changes update Kt/R/L together, and added mass changes carrier mass and inertia with readback verification. This course importer retains native joint armature and does not add rotor inertia. A packaging/rotor-inertia co-design requires a more complete mechanical model. Actual motor RPM was not exported in existing public recordings: the UI's calculated unloaded RPM must not be read as measured shaft speed. Candidate importer validation is tested locally; these new combined candidates have not been run in Isaac.

## September 12: challenge course and insertion recovery

`challenge_course.py` authors a 19 m route with stairs, a 22 cm gap, a 1.05 m
wide raised crossing, seeded rubble, an 8 degree side slope, and a physical
70 N body-local Y disturbance after reaching x=17 m. At 50 Hz the push spans
13 steps (0.26 s, 18.2 N s). The native rough-terrain policy receives a world-forward
velocity target with lateral and yaw feedback. Each variant uses the same navigation
rule, geometry, seed and checkpoint; feedback commands can differ with motion.
Success requires the route corridor, no reset, x>=19 m, and a one-second stop.
The bridge corridor is |base y|<=0.35 m. This is a step-over gap, not a jump policy.

Isaac Lab 2.1's ray caster reads only its first mesh. The authored invisible
`SensorSurface` concatenates exactly the colored physical collision zones;
`test_challenge_course.py` verifies this equality. It has no collision API and is
excluded from rendering. The earlier `challenge-stock-01` development rollout had
an incomplete height scan and is retained under results, excluded from comparisons.

```bash
./isaaclab.sh -p /workspace/atlas-suite/eval_anymal.py --headless \
  --course --course-layout challenge --record --num-envs 1 --seconds 45 \
  --checkpoint /workspace/atlas-suite/anymal-rough-stock.pt \
  --output /workspace/atlas-suite/challenge-stock
# Add --atlas --design kv_high (or nominal, gear_low, torque_low,
# mass_added_double), or --atlas --design-file candidates/torque_focused.json.
python isaac/suite/audit_challenge.py path/to/exported/replay
```

Public `lab/data/challenge` preserves original evaluations in `source-result.json`,
measured body poses and telemetry, independent audit results, and full videos.
The extra audit checks base clearance above the pit floor and finite-difference
speed during the final second; those clearance thresholds are explicitly post-run.
All design results are single-seed development examples. The torque-focused
candidate changes multiple parameters together, so it is not a single-axis ablation.
Native ANYdrive also changes the actuator/controller, so it is a separate reference.

The paired Factory insertion trials use checkpoint
`fcdc6014741a8b4ef9173b5d92471bbe979522873f1294013df6d9c6be6c2559`,
seed 201, 16 environments, one episode. Both first recorded initial poses match
exactly. Unperturbed: 16/16 sustained successes. With `--push-force 2`: 13/16.
The nominal [0.4,0.7) second command window yields five control steps at 15 Hz,
1/3 second and 2/3 N s. It acts on the held asset's local Y axis. It is an applied
external force, not a measured contact force. Environment 0, episode 0 was selected
before outcomes; both examples succeed. All 16 outcomes accompany each recording.
The task still prepares the grasp, and these experiments do not test actuator
variants or physical LeKiwi hardware.

A separate GearMesh probe transfers the same insertion checkpoint without
retraining: 4/16 sustained successes; preselected environment 0 fails. The
exporter includes both physical flanking gears (17 recorded bodies total).
`gear-transfer-02` repeats the evaluation with complete geometry capture and
matches the initial 4/16 result. This is not the dedicated GearMesh training run.
That training continues in the existing bounded assembly supervisor, followed by
NutThread. Both keep their iteration caps and the RunPod hard deadline.

Publication uses `.github/workflows/pages.yml`: deploy the reviewed `lab/` build,
article and research artifacts, excluding the duplicate `lab-src/` source assets.
Run `npm --prefix lab-src run typecheck` and `npm --prefix lab-src run build`
before committing. GitHub Pages remains at the original repository URL.
