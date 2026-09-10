# Allegro precision-control experiment

This is an Isaac Lab policy experiment. The short fine-tuning pilot regressed and its weights are not used in the demo. The training command below is an experimental reproduction recipe, not a validated way to improve the hand. The previous video counted brief orientation crossings. The new acceptance test requires a continuous 0.5-second hold within 0.1 rad, angular speed below 0.5 rad/s, linear speed below 0.05 m/s, and position error below 0.12 m.

Each 60-second evaluation has six 10-second stages: initial orientation, Y45, Y90, X90, Z90, and return with a physical 0.08 N cube-local-x pulse at 51–51.2 seconds. All six holds and zero drops are required for a complete sequence. Commands never overwrite the physical cube pose. Video selection cannot replace multi-seed evaluation.

## Reproduction

Tested with Isaac Lab 2.1, Isaac Sim 4.5, RSL-RL, and an L40S RunPod GPU. Supply a compatible 72-observation, 16-action Allegro checkpoint; the checkpoint and empirical observation normalizer are loaded together. The original checkpoint SHA256 is `295e2fdb2868069afd690e9b785815ef7256586e021d06e60107e32b0885282f`. Weights are not embedded in this source directory.

From this directory, with `ISAACLAB` pointing to the Isaac Lab installation:

```bash
export PYTHONPATH="$PWD/atlas_actuators_ext"
"$ISAACLAB/isaaclab.sh" -p eval_scripts/train_hand_precision.py \
  --checkpoint /path/to/model.pt --output /path/to/new-run \
  --num_envs 128 --iterations 200 --physics-substeps 4 \
  --finger-damping 0.005 --adaptive-curriculum \
  --initial-std 0.03 --entropy-coef 0 --seed 42 --headless

"$ISAACLAB/isaaclab.sh" -p eval_scripts/train_hand_precision.py \
  --checkpoint /path/to/new-run/final.pt --output /path/to/new-evaluation \
  --eval-only --eval-seeds 201,202,203 --physics-substeps 4 \
  --finger-damping 0.005 --headless

python tools/summarize_precision.py /path/to/new-evaluation/evaluation.json \
  --output /path/to/summary.json
PYTHONPATH=atlas_actuators_ext python -m pytest tests -q
```

The physics rate is 480 Hz; policy actions remain 30 Hz. The explicit finger PD damping is 0.005 Nms/rad, with inherited 3 Nm/rad stiffness and startup gain randomization. These are experimental controller settings, not identified hardware gains. Original thermal, current, voltage, and torque limits remain active. The warm-start fine-tuning adds stable-hold rewards, an overshoot penalty, action-change cost, and requested-versus-delivered torque cost. The adaptive curriculum begins at ±10 degrees and expands by 5 degrees after each sustained hold.

## Numerical and experimental limits

The original implicit-controller damping of 0.1 was inherited by an explicit torque controller. A local unconstrained mass-matrix diagnostic suggests a damping-only integration bound around 0.000275 seconds for that damping. The 480 Hz timestep is 0.002083 seconds. Reducing damping to 0.005 raises that diagnostic bound to about 0.00550 seconds. This calculation does not certify stability with contacts, stiffness, current lag, and torque saturation; observed task performance remains the deciding test.

Development seeds 101–103 were used for pilots. Reserved comparison seeds are 201–203, with 32 environments per seed. Same-seed baseline and candidate tests use the same startup settings. Comparing 120 Hz to 480 Hz or changing damping also changes the controller/simulator; those differences must not be attributed entirely to learning. Training is warm-start fine-tuning, not independent training from scratch.

`--terminal-hold` is a separate experimental controller: it retains previous finger targets near the goal and releases them when orientation error exceeds a wider threshold. It does not change physical state or motor limits. Its results must be labeled separately from the learned policy alone. `--goal-rate` is another optional experiment and is disabled in the standard benchmark. Success is always measured against the final stage orientation.

Runs write configurations, checkpoint hashes, raw per-trial outcomes, and completion status. A short `--eval-seconds` run is a diagnostic prefix, not a completed six-stage evaluation. Consult `RESULTS.md` for measured outcomes.

The inherited Isaac Lab orientation sampler uses `quat_x(a) * quat_y(b)`, with two random angles. Despite its docstring describing uniform 3D orientation sampling, that construction is a restricted target family. The new curriculum uses arbitrary axes and angles relative to the current physical orientation. This broadens target support, but its sampling distribution is intentionally curriculum-driven, not uniform over SO(3). Hard fixed-axis sequences therefore test more than the old instantaneous-hit objective.

The actor retains 72 observations for checkpoint compatibility. It does not receive explicit winding-temperature observations. Actuator awareness in this pilot comes from the real actuator dynamics and a torque-shortfall penalty; it is not a new temperature-conditioned network.

The same 32 startup physical randomizations are reused across the three reset seeds. The 96 trials therefore do not represent 96 independent actuator configurations. See the raw per-seed outcomes rather than treating every stage as an independent statistical sample.

The legacy `torque_shortfall_fraction` metric is the fraction of joints whose absolute requested-versus-applied torque difference exceeds 5% of the request magnitude, using a 0.1 Nm denominator floor. It includes lag/overshoot as well as under-delivery; the interface labels it torque mismatch.
