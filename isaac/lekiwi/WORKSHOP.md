# LeKiwi workshop challenge: build and test a gearbox

Status: proposed task specification; no LeKiwi hardware run or policy validation yet.

## The demonstration

One continuous task: drive to a parts tray, pick a gear, dock at a workbench, insert two gears onto shafts in a fixture, engage a removable crank, rotate the input and verify that the output gear turns, then deliver the finished assembly. A deliberately offset part supplies a recovery trial: withdraw, observe again, adjust the approach, and retry. Separate the recovery trial from undisturbed trials in reporting.

The first physical version uses large, chamfered parts and a bench fixture. Size and mass must be chosen after measuring the delivered gripper, reach and motor configuration. No tight-fit industrial precision claim follows from this version. The leader arm is for teleoperation; coordinated two-arm assembly requires a second actuated follower and its controller. The fixture is explicit in both the real and simulated scenes.

## Four connected experiments

| Experiment | Controlled challenge | Evidence |
| --- | --- | --- |
| Mobile manipulation course | Different tray locations, approach distances, payloads and cycle speeds | Complete task, drops, docking error, cycle time, interventions |
| Workshop and recovery | Gear insertion, crank pickup and operation; controlled pose offsets | Stage completion, retry count, actual gear motion, successful recovery |
| Actuator comparison | Fixed policy with different motor models in Isaac Lab | Success, tracking error, electrical energy, saturation, simulated winding temperature |
| Design optimization | Propose winding/Kv, gear ratio, current limit and actuator mass/geometry combinations | Feasible-design leaderboard; separate fixed-policy and retrained-policy results |

A working gearbox test requires visible output rotation in the expected direction and ratio, not merely apparently aligned meshes. The acceptance tolerance, shaft dimensions and fixture tolerances must be fixed before held-out trials once the physical parts are measured.

## Atlas experiment

1. Record the stock LeKiwi across payload, reach and speed conditions. Log timestamps, commanded and measured joint positions, camera frames and all supported servo diagnostics with model/firmware identifiers.
2. Fit an effective stock-servo model using one data split. Validate its motion predictions on different payload/trajectory trials. Compare with the same planner and policy under an ideal actuator model. Do not present a BLDC FOC model as the measured internal model of an unidentified stock servo.
3. Reproduce the physical task in Isaac Lab / PhysX. Keep fixture constraints, part dimensions, grasp preparation and resets explicit. Use the hardware experiment to test model fidelity before making design predictions.
4. Substitute Atlas design-derived actuator models in simulation. Kv changes must propagate to Kt and winding parameters; gearing affects the torque-speed envelope and reflected rotor inertia. Motor mass and its mounting position change physical link mass, COM and inertia. Packaging, electrical supply and mechanical limits are feasibility constraints.
5. Evaluate each design with a shared frozen policy first. Retraining is a separate experiment with equal training budgets. Split optimization/development and held-out trials; include every failed run in the leaderboard. Rank success first, then compare time, energy, mass and temperature without conflating them into an unexplained score.
6. Atlas hardware performance remains predicted until an actual compatible actuator assembly is installed and tested. A servo torque-limit register is not a winding or gearbox change.

## Viewer

Real camera video, measured robot-pose replay, and Isaac prediction share a timestamped timeline. A distinct comparison view shows simulated Atlas alternatives, not a synthetic overlay claimed as measured hardware motion. Stage labels: collect → dock → insert → recover if needed → crank test → deliver. Show the active design, trial outcome and any human intervention.

Measured and inferred quantities must be distinguishable. Stock servo load/current signals are not calibrated contact-force measurements; real force plots require an instrumented fixture or force sensor. Servo temperature is not automatically winding temperature. Supply energy needs calibrated voltage/current measurements and a declared system boundary; software estimates remain labelled estimates.

## Delivery stages

- Hardware arrival: inventory motor models, voltage variants, cameras and leader/follower roles; calibrate, teleoperate, record synchronized commanded/measured motion and a simple pick/insert sequence.
- Bench milestone: autonomous fixture insertion with random starts, explicit retries, and a functional gear-motion check; report every evaluation trial.
- Full sequence: mobile collection, docking, assembly, recovery and delivery without hidden resets or operator corrections.
- Atlas milestone: validated stock-servo simulation, fixed-policy design sweep, then budget-matched retraining and optimizer leaderboard.

The full autonomous workshop is not promised for the first day. Nut threading and two-follower manipulation are later extensions; native Franka task checkpoints cannot simply be deployed on LeKiwi.

## References

- Partabot LeKiwi listing: https://partabot.com/products/lekiwi-robot-arm?variant=43928617549939
- LeRobot LeKiwi hardware and demonstration workflow: https://huggingface.co/docs/lerobot/lekiwi
- DexMimicGen, demonstration-based multi-stage manipulation: https://dexmimicgen.github.io/
- Native Isaac Lab environments: https://isaac-sim.github.io/IsaacLab/v2.1.1/source/overview/environments
- ANYmal Parkour, inspiration for a measurable multi-stage course: https://arxiv.org/abs/2306.14874
