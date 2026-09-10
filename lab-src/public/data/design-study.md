# Design study results — 2026-09-10

All three Isaac Lab sweeps completed: 18 designs × 32 physical replicas × 120 seconds, for 1,728 trials. One policy is held fixed per robot. These are simulated sensitivities, not hardware validation or independently retrained design optima.

## Walking and balance

ANYmal's 1 m/s stage (t=40–120 s, including the physical shove and normal episode resets):

- Reference: 0.981 m/s, estimated 147.2 W.
- Gear 6:1: 0.981 m/s, 186.2 W (+26.5%).
- Gear 12:1: 0.981 m/s, 136.0 W (−7.6%).
- Add 6 kg across actuator carriers: 0.945 m/s, 162.7 W.
- Kv rewinding and a higher peak-current ceiling barely change this walking task. Additional available torque is not automatically useful.

The 2 m/s ANYmal stage repeatedly fails across all variants with this published walking policy. That failure is preserved in the full results, rather than presented as successful sprinting.

G1 with the corrected explicit-drive arm gains completes the full reference trial without falls in all 32 replicas. Adding 9.25 kg across actuator carriers causes 1,031 falls in aggregate across the same-duration trials (automatic resets permit repeated failures). This equal-per-joint mass intervention includes finger joints; it is not a proposed production actuator package.

G1's absolute motor-power estimates are **not timestep-converged**; see the numerical check below.

## Dexterity

- Reference: 40.92 goals/hand/minute, 2 drops across all 32 two-minute trials.
- Double thermal resistance: 20.62 goals/minute, peak winding temperature 107.7°C versus 69.0°C reference.
- Add reflected rotor inertia: 25.62 goals/minute.
- Add 160 g housing mass: 40.47 goals/minute.
- Higher Kv: 40.92 goals/minute, estimated input 141.2 W versus 151.2 W reference, but 8 drops versus 2.
- Lower Kv or a 50% higher peak-current limit destabilizes this fixed policy. Such interventions require controller retuning/retraining before ranking their achievable capability.

The thermal model continues operating with a continuous-current floor; a high simulated winding temperature is not a claim that real hardware would safely do so.

## Artifacts

- `anymal_walking_effects.png`, `g1_walking_effects.png`: walking-stage speed, motor input and temperature.
- `*_design_effects.png`: complete design sweeps, including failures.
- `*_time_effects.png`: measured trajectories through the challenge.
- `motor_tradeoffs.png`: FOC torque/speed reference curves at 25°C and 100°C; theoretical model curves, separate from task measurements.
- `design_matrix.csv`: actual parameter values and outcomes for every robot/design/actuator group.
- `uncertainty.json`: paired bootstrap intervals across startup replicas, conditional on the fixed policies.
- `video_anymal_gear_clean/comparison_share.mp4`: actual paired reference/6:1 walking and force-pulse recording.
- `video_g1_mass_clean/comparison_share.mp4`: actual paired reference/added-mass G1 recording.

All videos are illustrative one-replica trials, separately recorded with their own declared command schedule. The full 32-replica results determine the reported aggregate effects.

Validation: 36 local tests pass; one independent reference test is skipped because the separate reference implementation is absent. The reusable G1 explicit-drive task configuration was also loaded successfully in Isaac Lab.

## Numerical sensitivity check

Four G1 replicas per design, 20 seconds at 1 m/s, identical policy rate.

| Physics timestep | Reference estimated W | Higher-Kv estimated W | Reference peak °C | Higher-Kv peak °C | Falls |
|---|---:|---:|---:|---:|---:|
| 5 ms | 851.7 | 558.2 | 53.50 | 39.00 | 0 for both |
| 2.5 ms | 984.6 | 694.6 | 53.50 | 39.00 | 0 for both |

Halving the timestep shifts estimated power by 16–24%. Lower heating and the direction of the power difference persist, but absolute electrical efficiency is not converged. Small tracking-score rankings also change. The 120-second G1 sweep uses 5 ms; interpret it as a conditional simulator/controller sensitivity study. Motor power sampled at the control rate can be sensitive to rapid torque/velocity oscillations. A future engineering-grade energy study should integrate power at physics-substep rate and establish timestep convergence.
