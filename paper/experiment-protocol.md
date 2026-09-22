# Atlas design-search experiment protocol

**Status: proposed; no search or final test has been executed under this protocol.** Numerical budgets below are planning choices to be frozen in a versioned configuration before execution. No GPU job is launched by preparing this document.

## Claim to test

Atlas selects actuator designs that improve completion of unseen tasks under the same mass and power limits. Separately test whether calibrated electrical inputs change the selected design and its held-out performance.

## Experiment matrix

| Study | Conditions | Control | Primary result | Status |
|---|---|---|---|---|
| Electrical intake | Supplied nominal vs mj5208 controller calibration | Preserve units, conventions and raw-log hash | Parameter shifts and conditional model implications | Complete |
| Hand adaptation | Simplified vs motor-model fine-tuning | Common initialization, explicit PD, reward, budgets and 3 paired training seeds | Cold motion success, with stricter contact outcome separate | Complete |
| Design-search quality | Atlas, uniform feasible random search, conventional mixed-variable optimizer, nominal reference | Same feasible designs, task instances and evaluation budget | Held-out completion of selected design | Pending |
| Model value | Same selection method using simplified, nominal-electrical and calibrated-electrical models | Same search budget; shared reference evaluation and uncertainty scenarios | Selected-design changes and paired held-out performance | Pending |
| Controller adaptation | Common frozen policy vs equal-budget retraining for selected designs | Same initial policy and PPO settings; 3 policy seeds | Whether adaptation changes selected-design performance/ranking | Pending |
| Mechanical attribution | Full design vs matched winding, gearing and placement ablations | Same controller comparison and evaluation tasks | Which changes explain the observed outcome | Pending |
| Thermal sensitivity | Declared cooling ranges and separate hot-start stress cases | Same selected designs, workloads and loss accounting | Sensitivity of ranking to thermal assumptions | Pending |

## Proposed sampling and budget

- Start with the locomotion course; add one arm workload for a cross-task claim. Keep the pen experiment as the completed control case.
- Use **5 independent search seeds**, each allowed **64 candidate evaluations**. A candidate evaluation uses **32 common development task instances**. This is a planning budget, not a power calculation or evidence that 64 candidates are enough.
- Provide **16 separate validation instances** for a fixed shortlist of four candidates per method and search seed. Select exactly one design per search run. The validation allocation is identical across methods and counted in total compute.
- Generate **64 untouched test instances** before the search. Include both fresh seeds within the development distribution and separately labeled withheld obstacle/load combinations. No final-test result is used for selection or tuning.
- For adaptation, retrain each selected design and the nominal reference with **3 common policy seeds** and the same transition budget, fixed after a development-only feasibility pilot. Report the chosen budget and wall-clock cost. Earlier hand-study budgets do not automatically transfer to locomotion.
- Preserve the full search histories. Uniform random search samples the same feasible family; declare its distribution. Specify the conventional optimizer, its initialization and all hyperparameters. Document Atlas's actual selection method and any training data, surrogate, or pretrained component before claiming an algorithmic contribution.

## Physical design contract

Each candidate records winding identity/specification, convention-labeled Kv and Kt, phase resistance, Ld/Lq, drive limits, gear ratio and efficiency, motor/transmission mass, rotor inertia, placement and thermal assumptions. Current importer bounds are implementation limits, not a validated feasible hardware family.

Define one shared actuator mass ceiling, supply budget and packaging region before search. Account for all motor and transmission mass. A feasible rewind couples turns, wire/current limits, resistance and inductance. Placement changes centers of mass and inertia tensors. Gearing changes reflected rotor inertia and transmission properties. Read back these values from the simulated articulation and archive them. Validate imported configurations before rollout.

The mj5208 measurements apply only to their identified motor family. A scaled or virtual embodiment is explicitly modeled and labeled. Other robots require their own parameter provenance; the calibration cannot be copied into stock actuator specifications.

## Scoring and failure accounting

Primary score is completion rate within task-specific time, corridor and stability rules fixed before search. Use completion first and predeclared tie-breakers such as successful-trial time and copper loss. Retain the multi-objective results rather than hiding every tradeoff in one weighted score.

Report resource-limit violations and incomplete tasks separately from completed-task energy/time summaries. Candidate-caused infeasibility or numerical failure counts against its evaluation budget. Infrastructure failures permit a logged rerun of the same configuration and seed; replacements cannot depend on whether a result was favorable. Predeclare the retry limit.

The primary comparison is paired at the independent search-run level. Show all five outcomes and paired differences. Task trials and policy seeds are nested replications, not independent searches. Any uncertainty interval must respect that hierarchy and disclose the low search-run count. Negative results remain reportable.

## Unseen test and film

Freeze the selected design, policy and evaluation code before opening test outcomes. Preselect the video trial and retain failures. Report aggregate test results beside the clip. The original and revised design use the same camera, starting state, challenge and budget for the development comparison. The final unseen test remains distinct.

The film is a view of the experiment ledger: **candidate grid → physical design revisions and motion → initial/revised outcome → untouched test**. A successful edit requires the corresponding measured simulation outcomes; a storyboard is not a result.

## Required experiment record

One versioned manifest must map each design ID to its parameter provenance, code/runtime version, search method/seed, evaluation split, task seeds, policy/checkpoint hash, physical mass/inertia readback, compute budget, full outcome table, torque/RPM/current traces, assumed thermal model, rendered video and body-state replay. Include external asset/policy licenses and human interventions. Archive failed and rejected candidates alongside selected designs.
