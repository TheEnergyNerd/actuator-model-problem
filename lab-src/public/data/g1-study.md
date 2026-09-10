# G1 actuator sensitivity with the new walking policy

18 designs x 16 paired replicas x 20 seconds; command 0.7 m/s. Fixed Walking-v3 policy, 2.5 ms physics, 50 Hz actions. No external push during this test.

These short, cold-start trials measure transfer of one policy across motor designs. They do not measure the best gait achievable after retraining each design or long-duration thermal endurance. Falls are counted across episode resets. Temperature is a motor-model estimate. Absolute electrical-power convergence remains unresolved, so no efficiency ranking is claimed here.

| Design | Added mass (kg) | Body speed (m/s) | World-x speed (m/s) | Slip (cm/s) | Contact phase | Falls | Model peak (C) |
|---|---:|---:|---:|---:|---:|---:|---:|
| nominal | 0.00 | 0.639 | 0.605 | 3.5 | 95.7% | 0 | 53.5 |
| kv_low | 0.00 | 0.638 | 0.583 | 3.0 | 94.8% | 0 | 75.2 |
| kv_high | 0.00 | 0.625 | 0.612 | 2.9 | 95.8% | 0 | 39.0 |
| torque_low | 0.00 | 0.625 | 0.614 | 2.9 | 95.8% | 0 | 39.1 |
| torque_high | 0.00 | 0.638 | 0.583 | 3.0 | 94.8% | 0 | 75.5 |
| gear_low | 0.00 | 0.624 | 0.609 | 2.8 | 95.8% | 0 | 53.5 |
| gear_high | 0.00 | 0.638 | 0.584 | 2.9 | 94.8% | 0 | 53.5 |
| voltage_low | 0.00 | 0.639 | 0.605 | 3.5 | 95.7% | 0 | 53.5 |
| voltage_high | 0.00 | 0.639 | 0.605 | 3.5 | 95.7% | 0 | 53.5 |
| copper_low_R | 0.00 | 0.638 | 0.602 | 3.6 | 95.7% | 0 | 40.1 |
| copper_high_R | 0.00 | 0.639 | 0.610 | 3.4 | 95.7% | 0 | 76.2 |
| cooling_better | 0.00 | 0.639 | 0.605 | 3.5 | 95.7% | 0 | 51.7 |
| cooling_worse | 0.00 | 0.639 | 0.606 | 3.5 | 95.7% | 0 | 54.5 |
| mass_added | 9.25 | 0.593 | 0.581 | 3.4 | 93.9% | 0 | 53.5 |
| mass_added_double | 18.50 | 0.554 | 0.365 | 15.1 | 82.0% | 0 | 53.5 |
| rotor_heavier | 0.00 | 0.639 | 0.599 | 2.9 | 94.9% | 0 | 53.5 |
| high_gear_with_mass | 9.25 | 0.608 | 0.473 | 2.8 | 95.2% | 0 | 53.5 |
| higher_current_with_cooling | 9.25 | 0.615 | 0.497 | 4.2 | 95.8% | 0 | 61.8 |

The CSV includes phase-peak Kv, Kt, leg gear ratio and nominal peak leg-joint torque. Full per-group motor parameters and per-replica measurements are in results.json. Housing mass is added to actual joint parent links with corresponding inertia; a 0.25 kg increment at all 37 joints adds 9.25 kg in total. Kv changes use a consistent rewind model, including the reciprocal Kt change, rather than changing a label alone.

Checkpoint SHA-256: `16cd9f934f5129735d1be9c07386bc0f96ed0c07d93173c5af99cc898ab7783c`.
