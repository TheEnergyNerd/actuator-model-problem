# atlas_actuators — Isaac Lab extension

Drop the Atlas design tool's *real* actuator physics into Isaac Lab, so RL
policies train against the torque-speed envelope, inverter voltage limit,
field weakening, and thermal derate the hardware actually has.

## Install (editable, from the motordesign repo checkout)
```bash
# inside your Isaac Lab python env (e.g. /isaac-sim/python.sh -m pip ...)
pip install -e isaac/exts/atlas_actuators
```

## Two fidelity tiers, one design dict
| tier | cfg | physics | use |
|---|---|---|---|
| envelope | `AtlasActuatorCfg` | τ-ω/thermal clamp (resistive voltage model) | cheapest; large-scale RL |
| **FOC** | `AtlasActuatorFOCCfg` | field-weakening voltage ellipse (resistive margin), current circle + thermal derate, first-order current lag; optional true dq substepping (`mode="dq"`) | FOC-correct ceiling — the envelope is optimistic near base speed (13.75 vs 12.44 N·m in our reference design) |

Both take their parameters from an Atlas design run's `controller` dict — the
same numbers that drive the emitted MCU firmware (sim == firmware, 6e-7).

## Use
```python
from atlas_actuators import AtlasActuatorFOCCfg, foc_cfg_kwargs

robot_cfg.actuators = {"legs": AtlasActuatorFOCCfg(
    joint_names_expr=[".*"], stiffness=120.0, damping=6.0,
    **foc_cfg_kwargs(controller, gear_ratio=9.0, V_bus_V=48.0))}
```

## Example tasks + training
```bash
# registered by `import atlas_actuators.tasks`
#   Isaac-Velocity-Flat-Anymal-C-Atlas-v0       (envelope)
#   Isaac-Velocity-Flat-Anymal-C-AtlasFOC-v0    (FOC)
/isaac-sim/python.sh scripts/train.py --actuator atlas-foc --num_envs 4096
/isaac-sim/python.sh scripts/train.py --actuator ideal      # the baseline
```
The reward/behaviour delta between `ideal` and `atlas-foc` is the sim-to-real
gap this extension closes.

## Verification status (honest)
- **foc_core (batched physics): tested locally, 9/9** — fast path matches the
  scalar dq model (power balance 0.00%, firmware parity 1e-6) exactly across
  the speed range incl. field weakening; numpy==torch to 4e-14 at 4096×12.
- **Envelope tier: verified in-engine** on an L40S (Isaac Sim 4.5 + Isaac Lab
  2.1): boots, loads as ActuatorBase, drove an ANYmal-C (rendered demo).
- **FOC tier + tasks + train script: not yet run in-engine** — same
  import-guarded pattern as the verified envelope tier, but honest status is
  "engine-verification pending a GPU pod run".

## Known limits
- Editable/source install only (the envelope tier re-exports its canonical
  single-file implementation from `isaac/atlas_actuator.py`).
- Saliency (Ld≠Lq) defaults to an analytical topology split; exact values need
  the FEA d/q inductance sweep (documented in `dq_inductance.py`).
- No commutation/encoder-calibration model yet (bring-up effects).
