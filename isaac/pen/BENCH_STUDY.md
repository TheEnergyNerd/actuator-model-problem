# Bench-informed software study — protocol fixed before runs

This is an Isaac Lab software experiment, not hardware transfer validation.
The mj5208 controller calibration supplies electrical parameters for a **virtual
remote-drive hand design**. It is not a claim that mj5208 motors fit inside Sharpa.
Native hand geometry and link mass are retained; packaging, rotor inertia and
transmission dynamics are not modeled. See `bench_profile.py` for provenance.

## Hardware evidence and conversion

Raw log SHA256: `1e1b08b78c474780261213c34723f02d504bbc77c3140a81100e5b2ac133b4bc`.
The values match the provided summary: R=0.0650753979 ohm, Ld=27.0114475 uH,
Lq=45.2419470 uH, Kv=310.675047 RPM/V, 7 pole pairs. Scalar L is retained in
source provenance but is not substituted for either measured axis inductance.
The recorded tool is moteus 1.1.1, firmware
`6f063a90a97012dc9f6ccc420390c5a6cc55fac7`, ABI 65536.

The exact tool defines Kv as RPM per **peak line-line voltage**. Multiplying by
sqrt(3) gives phase-peak Kv; the pinned firmware's torque conversion then gives
Kt=0.0266192393 Nm per peak q-axis ampere. The handoff's 0.03073725 conversion
used a different convention. Neither number is measured shaft torque. Separate
Ld/Lq are preserved, including the model's reluctance-torque contribution.
Calibration resistance temperature and repeatability remain unspecified.

12 V, 2 A peak, 1 A continuous, 80% transmission efficiency and per-joint gearing
chosen to match native cold nominal torque ceilings are **assumptions**. Thermal
R=10 K/W, C=8 J/K, ambient/reference=25 C and linear current derating to 85 C
are unvalidated scenario settings. Requested 200 Hz calibration bandwidth is
approximated as first-order current lag; it is not a measured transfer function.

## Matched training and independent evaluation

- Same original pretrained policy, frozen normalizers, native prepared grasp,
  explicit PD gains, task objective and PPO hyperparameters in both conditions.
- Simplified training: fixed nominal torque ceiling. Motor training: calibrated
  electrical parameters plus explicitly assumed drive/transmission/thermal law.
- Three paired training seeds: 44000000, 44010000, 44020000. Each uses 256 prepared
  grasps. Seeds control initialization/settling, action sampling and temperature RNG.
- 256 iterations × 64 steps × 256 environments = **4,194,304 transitions per run**;
  six runs total. Eight times the prior per-run budget. Final checkpoint is fixed.
- Both start every episode at 25 C. No hidden randomized starting temperature.
  Normal within-episode motor heating remains modeled. Policy observations unchanged.
- Validation: seeds 45000000–45000031, checkpoints 64, 128, 256, cold motor plant.
  Report all curves. No hyperparameter adjustment or checkpoint selection from test.
  Longer training does not establish convergence: report whether validation is stable.
- Final fresh tests: seeds 46000000–46000127. Evaluate each final policy under
  fixed torque, cold motor and 100 C-start motor. Original frozen policy is a control.
  Training/evaluation use the same stated virtual design. Hot starts are a separate
  out-of-distribution stress scenario; they are not calibrated operating conditions.
- Primary outcome: motion success under the cold motor plant, paired by training
  seed and test grasp. Report all three seed results and mean, plus seed range.
  Do not treat 384 observations from three policies as 384 independent training runs.
- Secondary: full native contact gate, timing, drops, torque shortfall, simulated
  copper heat and temperature. All assumptions remain visible with the results.
- Display seed 46000000 and training seed 44000000, fixed in advance, regardless of
  outcome. Preserve the previous pilot, with a link to this separate study.

No retrospective actuator tuning to create a winner. A negative or small effect
is useful: it can mean this task rarely encounters a modeled constraint. Inspect
constraint activation before attributing score differences to actuator physics.
Do not interpret an internally consistent simulator as independent hardware truth.
