# Sharpa pen-policy reproduction

The first baseline reproduces the provided checkpoint from
[Dexterous Astra](https://github.com/jianglongye/dexterous-astra), pinned to
`a1c93b3ab814995b0c75bae51595417caa91c34b`. The policy is external reference work,
not a policy trained by Atlas. Reference source and weights are fetched separately.

Dependencies: Isaac Lab 2.3.2, Isaac Sim 5.1, Python 3.11, rsl-rl-lib 3.0.1;
official Sharpa assets at `0d447b6889e6d993758169dfc0aa75ee9f6ad8d7`.
Set `SHARPA_ROOT` to that asset checkout. No MuJoCo runtime is required.

```
./isaaclab.sh -p /path/to/isaac/pen/run_reference.py \
  --reference /path/to/dexterous-astra --out /path/to/new-run \
  --trials 32 --seed0 41000000 --headless --device cuda:0
python export_run.py /path/to/new-run --reference /path/to/dexterous-astra
```

The runner verifies the pinned source-file and checkpoint hashes, uses weights-only
checkpoint loading, and instruments evaluation to record every rigid body and
joint torque. Policy observations, actions and physics are unchanged. Trial 0 is
selected for the replay before seeing outcomes; all 32 outcomes are retained.
A prepared, settled grasp precedes evaluation. This is not a pickup demonstration.

The upstream independent penetration check and video renderer use MuJoCo. Atlas
instead reports the native PhysX contact separations and renders the recorded
body poses in Isaac. The resulting contact gate is explicitly distinct from the
upstream gate; initial penetration is not recorded by the reference evaluator.
Do not call this a complete reproduction of their acceptance result.

The initial baseline uses native Sharpa implicit position control. It does not
establish measured hardware torque, thermal limits, or performance under Atlas
actuator variants. First confirm the supplied skill, then compare matched
explicit controllers before introducing motor parameters and retraining.

For matched sensitivity runs, pass `--effort-scale 0.25` or
`--hand-mass-scale 1.5`. Both interventions apply after the same seeded grasp
settling; the latter scales mass and inertia together. The runner reads the
physical limits/masses back from PhysX and writes `intervention.json`.
This does not change the checkpoint or retrain the policy.

After `export_run.py`, run `style_replay.py RUN/replay/scene.json`, then the
shared `isaac/suite/render_recording.py --recording RUN --fps 60 --headless`
using Isaac's Python. It renders every recorded frame. The hand palette and
studio floor are presentation choices, not physical changes.

The baseline and two sensitivity runs are complete; see [RESULTS.md](RESULTS.md).

## Matched explicit motor-model experiment

Use `--model ideal`, `--model motor`, or `--model motor-hot` on otherwise identical
runs. `motor_model.py` disables native implicit drives after prepared-grasp settling
and uses identical explicit PD gains and cold stall ceilings in all three modes.
The cold/hot models use the existing FOC core's quasi-static references and first-order
current loop. They are assumed generic geared motors, not measured Sharpa hardware.
The current model does not simulate PWM, gearbox backlash, friction, or a calibrated
thermal network. No policy retraining is performed.

`export_run.py --trial 0` retains the preselected first attempt. The paired illustrative
view uses trial 1, the first ideal-model success selected before inspecting motor
outcomes. All 32 trials remain in evaluation.json. Motor traces sample the final
physics substep at 60 Hz; the actuator runs at 240 Hz. Copper energy accumulates at
240 Hz and is a model loss integral, not measured battery consumption.
