# Matched actuator-model fine-tuning

This experiment compares two copies of the same pretrained Sharpa pen checkpoint.
The upstream repository provides inference and evaluation code, but not the trainer
or reward implementation. `train_matched.py` implements a new Atlas PPO objective;
this is fine-tuning, not training from scratch or reproducing the original training.

Protocol fixed before training:

- Same initial checkpoint, explicit PD gains, geometry, mass, observations and actions.
- 128 prepared training grasps, seeds 43000000–43000127.
- 64 PPO iterations × 64 steps × 128 environments = 524,288 transitions per condition.
- Simplified condition: fixed torque clipping; motor condition: existing assumed
  FOC/current-lag/thermal model. Initial winding temperature is uniform 25–100°C
  on reset. The same temperature draws occur in the ideal condition but do not
  affect its torque. Other motor parameters remain fixed.
- Frozen observation normalization, actor LR 2e-5, critic LR 3e-4, fixed exploration
  standard deviation 0.2, clip 0.1, 3 epochs, minibatch 1024, gamma .99, GAE .95.
- A small penalty anchors actor outputs to the initial policy to limit forgetting.
- Final fixed-budget checkpoint; no checkpoint selection using test results.
- Evaluate the original, simplified-fine-tuned and motor-fine-tuned policies in
  fixed-torque, cold-motor and 100°C-start conditions on seeds 42000000–42000063.
  These seeds do not overlap the previously published development or training seeds.

Reward: supported forward rotation (capped per step), a quiet supported hold after
three rotations, a success bonus and a drop penalty; small action, linear velocity,
palm-force and tilt penalties. The reward is a learning surrogate. Final outcomes
are scored using the unchanged motion gate and separately reported native contact
gate. No forces act directly on the pen. The wrist remains fixed.

Resetting restores a native prepared grasp and clears current and copper-energy
states; temperatures are resampled. The thermal state is not added to policy
observations, so the comparison preserves the same actor architecture. Prepared
training grasps repeat between episodes; generalization is tested on separate seeds.
One training seed per condition is a pilot, not evidence of a reproducible advantage
across training seeds. Assumed motor parameters do not establish hardware accuracy.
