# Allegro precision-control results

Six fixed rotation stages, including a physical push. Each needs 0.5 continuous seconds within 5.7°, angular speed below 0.5 rad/s, linear speed below 0.05 m/s, and position error below 0.12 m. All six holds and zero drops are required for a complete sequence. Each row uses 96 trials across seeds 201–203, with 480 Hz physics and 30 Hz policy actions.

All four matched comparisons completed. The precision video uses the original policy with terminal hold; fine-tuned weights are not used in that recording.

| Policy/controller | Complete sequences | Stable stages | Drops | Mean error |
|---|---:|---:|---:|---:|
| Original policy · damping 0.1 | 0/96 | 89/576 | 1 | 35.1° |
| Original policy · damping 0.005 | 0/96 | 87/576 | 4 | 33.1° |
| Original policy + terminal hold · damping 0.005 | 0/96 | 227/576 | 5 | 37.8° |
| Fine-tuned policy · damping 0.005 | 0/96 | 3/576 | 19 | 66.9° |

The full 60-second recording uses a separately fixed seed 42, not a selected successful evaluation environment. It completed 2/6 stages, had zero drops, and did not pass the complete sequence. Its longest continuous qualifying hold was 6.47 seconds. Video and measured 3D poses contain 1,800 synchronized frames.

## Interpretation

Changing damping alone did not improve overall stable-stage success in this comparison. The terminal controller and policy fine-tuning are separate interventions. No instantaneous orientation crossing counts as a stable hold. Thermal, torque, current, and voltage limits remain active. The experiment does not establish hardware performance or Rubik-solving ability.

The original training target sampler uses X-then-Y rotations. The new curriculum expands arbitrary-axis rotations only after sustained holds. A short fine-tune therefore changes both the objective and target distribution; failure on harder rotations must be reported rather than hidden by a favorable clip.

Raw outcomes include each trial’s stage flags, maximum continuous dwell, drops, orientation error, torque shortfall, winding temperature, and checkpoint hash. Development pilots are retained in the source repository separately from these matched comparisons.

## Decision

Terminal hold increased stable stages from 87/576 to 227/576 under matched controller settings, but complete sequences remained 0/96. Mean orientation error increased from 33.1° to 37.8° and drops from 4 to 5. This is an improvement in individual holds, not overall reliable dexterity.

The fine-tuned candidate achieved 3/576 stable stages, 0/96 sequences, and 19 drops. It was rejected and is not used by the demo. The precision curriculum and runner are retained as experimental code, with this failure documented.
