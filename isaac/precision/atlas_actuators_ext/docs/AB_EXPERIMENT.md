# A/B experiment: what actuator model you train against decides what you deploy

**Date:** 2026-07-08 · **Hardware:** RunPod L40S, Isaac Sim 4.5 + Isaac Lab 2.1 ·
**Cost:** <$1 GPU time · **Task:** `Isaac-Velocity-Flat-Anymal-C` (stock velocity
tracking, flat terrain), PPO via rsl_rl, 4096 envs, 300 iterations, single seed.

## Setup

Two identical trainings, differing ONLY in the actuator model inside the sim:

- **Policy A** — stock task: ANYmal-C with `ANYDRIVE_3_LSTM_ACTUATOR_CFG` —
  **ETH's actuator net**, the LSTM learned from dyno data of ANYbotics' real
  ANYdrive. (The field's gold standard — for *that* hardware.)
- **Policy B** — same task, actuators swapped to `AtlasActuatorFOCCfg`: the
  physics model of *our designed actuator* (Kt=0.25, 55 A, 9:1, 48 V — FOC
  voltage ellipse, field weakening, thermal derate), generated from the design.

Cross-evaluation: both policies run in the **Atlas-FOC world** (proxy for "the
robot you actually built"), 256 envs × 1500 steps (~260 episodes each).

## Results

| policy ↓ evaluated in → | own training world | **Atlas-FOC world** |
|---|---|---|
| **A** (trained on ANYdrive net) | **+24.5** rew, 3.5% falls | **−243.9** rew, 1.5% falls |
| **B** (trained on Atlas FOC)    | — | **+2.9** rew, 3.0% falls |

- **The gap: ~247 reward points** between matched and mismatched training,
  in identical evaluation physics.
- Policy A rarely *falls* — it *underperforms*: its gait demands torque at
  speeds where the real motor's voltage budget can't deliver it, so tracking
  collapses and effort penalties accumulate. This is what real sim-to-real
  failure usually looks like: not explosions, degradation.
- In-engine sanity: the FOC actuator clamped applied torque at **111.4 N·m**,
  exactly the analytical envelope of the design (Kt·I_peak·G·η).

## Videos (`~/Downloads/atlas_ab_experiment/`)

- `A_policy_in_IDEAL_world.mp4` — the sim promise (walks well)
- `A_policy_in_FOC_world.mp4` — the same policy meeting real actuator physics
- `B_policy_in_FOC_world.mp4` — the actuator-aware policy in the same world

## Why this matters (the platform claim)

The only established fix for this gap is the actuator-net method: collect dyno
data from hardware that already exists, fit a model, retrain. Policy A *was*
trained on such a model — just for different hardware, and it did not transfer.
Atlas emits the equivalent faithful model **from the design itself, before the
motor is built** — same source of truth that generates the drive firmware
(sim == firmware parity 1e-6).

## Honest limitations

- Single seed, 300 iterations, default PPO hyperparameters; B's PD gains
  (stiffness 120, damping 6) untuned — B's absolute score would rise with
  tuning + longer training. The *gap*, not B's absolute number, is the result.
- The FOC world is a physics proxy for reality, not reality; the remaining
  step is a hardware round-trip on a bench drive.
- Rewards are comparable across worlds (identical reward function; only the
  actuator physics differ).

## Reproduce

```bash
# on an Isaac Lab pod with this extension installed:
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Anymal-C-v0 --headless --num_envs 4096 \
  --max_iterations 300 --run_name ideal
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Anymal-C-AtlasFOC-v0 --headless --num_envs 4096 \
  --max_iterations 300 --run_name atlasfoc
# cross-eval (scripts/eval_policy.py):
./isaaclab.sh -p eval_policy.py --task Isaac-Velocity-Flat-Anymal-C-AtlasFOC-v0 \
  --ckpt <ideal_run>/model_299.pt     # → the −244
./isaaclab.sh -p eval_policy.py --task Isaac-Velocity-Flat-Anymal-C-AtlasFOC-v0 \
  --ckpt <atlasfoc_run>/model_299.pt  # → the +2.9
```
(Stock train.py/play.py need one added line — `import atlas_actuators.tasks` —
at the extension-template placeholder.)

---

## v2 (2026-07-09): equal budgets, locomotion-standard gains — policy B walks

v1's Atlas run used untuned PD gains (Kp 120/Kd 6) and 300 iterations; the
trained policy survived but visibly did not walk (caught in video review).
v2 fixes both: **Kp 85/Kd 2** (legged-gym convention) and **1500 iterations
for BOTH policies** (equal budgets).

| policy trained against | own world | Atlas-FOC world |
|---|---|---|
| ANYdrive actuator-net | 26.14 · 1.9% falls | 24.49 · **5.0% falls** |
| Atlas FOC             | — | **25.89 · 2.7% falls** |

Findings:
1. **Training against the designed actuator costs nothing** — B reaches parity
   with the hand-tuned stock baseline (25.89 vs 26.14, each in its own world),
   walking fully (frame-strip verified), on the physics that ships.
2. **Mismatch still costs**: the transferred policy loses reward and falls
   2.6× as often — on easy flat ground. (~260 episodes/cell; fall counts 5–13,
   so 2.6× is indicative.)
3. **v1 reinterpreted**: the catastrophic −243.9 was actuator mismatch
   *compounded* with tuning error — kept above as a documented record of how
   badly the stack degrades when actuation differs from what was trained
   against, and as an honest correction trail.

Training curves: B matched A's learning almost exactly once gains were sane
(B final train reward 23.60 vs A 23.43). Cost of v2: ~$0.80.

---

## v4 (2026-07-10): where the difference actually lives

Instrumented evaluation (eval_traces.py) at a fixed 1 m/s cruise showed the
two policies are measurably identical in steady state: velocity tracking
0.97 vs 0.95 of command, torque saturation ~0% for both. The motor is
comfortably oversized for flat-ground cruising (consistent with the gear
sweep), so the fall/reward gap lives entirely in rare transients: recovery
from the task's random pushes.

Stress test (eval_push.py): same FOC world, both policies, harder shoves,
~300 episodes per cell.

| push magnitude | policy A (mismatched) | policy B (matched) |
|---|---|---|
| ±1.0 m/s (task default) | 5.0% falls | 2.7% |
| ±1.5 m/s | **43.5%** | **27.9%** |
| ±2.5 m/s | 84.6% | 70.9% |

The actuator-model mismatch is a robustness gap: invisible while cruising,
decisive when the policy must suddenly demand torque at speed to recover.
Checkpoints saved to ~/Downloads/atlas_ab_experiment/checkpoints/ (policies
reproduce with the fixed seed; ~1 MB each).

Synchronized-shove footage (2026-07-11): 16 robots per policy, identical
1.8 m/s shoves on a fixed 4 s clock, 15 s clips. In-clip falls: policy A 17,
policy B 10 (record_shove.py prints the count). This is the countable-on-
screen version of the robustness gap; clips in the blog post and
~/Downloads/atlas_ab_experiment/v3/.

---

## v5 (2026-07-11): hardened training does not close the model gap

Retrained BOTH policies with a widened disturbance curriculum (training
shoves ±2.0 m/s instead of the stock ±1.0; tasks
Isaac-Velocity-Flat-Anymal-C-HardPush-v0 / -AtlasFOC-Hard-v0), nothing else
changed. Final mean rewards ~21.5 / 21.6 (parity again, harder task).
Cross-eval in the FOC world (256 envs, 30 s episodes):

| push | A-hard (mismatched) | B-hard (matched) | (v4 standard-trained) |
|---|---|---|---|
| ±1.0 | 2.3% | 0.8% | 5.0 / 2.7 |
| ±1.5 | 5.9% | 3.1% | 43.5 / 27.9 |
| ±1.8 | 6.4% | 7.1% | — |
| ±2.5 | **16.5%** | **7.4%** | 84.6 / 70.9 |
| ±2.8 | 20.8% | 13.8% | — |
| ±3.0 | 26.1% | 18.6% | — |
| ±3.5 | 42.4% | 34.4% | — |

Hardening collapses absolute fall rates for both (that's what better
training buys) but the ordering and a ~2x gap at ±2.5 persist: the missing
information is in the actuator model, not the recipe.

Filming note: fixed FORWARD shoves stopped separating the hardened policies
on camera (a frontal shove is ridden out along the walking direction), so
record_shove.py gained --shove_mode random, matching the eval protocol; the
fixed env seed gives both policies the identical shove sequence. Footage at
3.5 m/s random: A falls 8, B falls 3 in 15 s (fall counters + flashes burned
in from falls.json). Checkpoints: ~/Downloads/atlas_ab_experiment/hard/film/.

---

## G1 humanoid (2026-07-11): the same experiment, two legs

Isaac Lab's stock G1 task drives every joint with effort_limit=300 N·m
(real G1 knee: 139 N·m peak per Unitree spec). Atlas variant
(Isaac-Velocity-Flat-G1-AtlasFOC-v0): one motor design geared three ways —
legs 12:1 (~148 N·m ceiling), feet/arms 2:1 (~25 N·m); stock PD gains,
armature, rewards. Both policies: seed 42, 1500 iters, 4096 envs. The stock
G1 task ships with push randomization DISABLED; kept that way for both
(eval_push/record_shove re-create the push event at eval time).

Training: stock survives everything by iter ~150; Atlas takes ~270 iters to
stand and oscillates 880–970 eplen (real torque limits keep collecting).
Final rewards 28.6 (stock world) vs 8.1 (Atlas world) — not comparable
across worlds; the cross-eval is the comparison:

Cross-eval in the Atlas FOC world (256 envs, 30 s):

| push | stock policy | Atlas policy |
|---|---|---|
| none | **100.0%** (4981/4983 fell) | **12.0%** |
| ±0.5 | 100.0% | 20.7% |
| ±1.0 | 100.0% | 46.1% |
| ±1.5 | 100.0% | 69.7% |

The stock humanoid policy cannot stand on real actuator physics at all —
every reflex was learned where torque is free. Footage (1.0 m/s shoves,
15 s, 16 robots/side): stock 179 falls vs Atlas 1. A quadruped pays the
mismatch as a robustness tax; a humanoid, dynamically stabilized every
millisecond, does not get up.

Checkpoints: ~/Downloads/atlas_ab_experiment/g1/film/policy{A,B}_g1.pt.
Blog post updated (figs 6, 9, 10 + three new annotated clips).

---

## G1 v2 (2026-07-12): trained to convergence, gait fixed

The 1500-iter Atlas G1 walked with a crouched gait: reward was still climbing
steeply at cutoff (survival dominates early training; gait-quality terms never
got their turn). Retrained BOTH policies at 3000 iters, stock recipe still
untouched. Stock converged unchanged (~27); Atlas reached 19.6 with an
upright, striding gait.

Cross-eval in the Atlas FOC world (256 envs, 30 s):

| push | stock 3k | Atlas 3k | (1.5k run: stock / Atlas) |
|---|---|---|---|
| none | 97.7% | **0.4%** (1 fall/256) | 100 / 12.0 |
| ±0.5 | 99.1% | 8.1% | 100 / 20.7 |
| ±1.0 | 100% | 41.7% | 100 / 46.1 |
| ±1.5 | 100% | 63.3% | 100 / 69.7 |

Footage at 1.0 m/s shoves: stock 117 falls vs Atlas 8 in 15 s. Checkpoints:
~/Downloads/atlas_ab_experiment/g1_3k/film/policy{A,B}_g1_3k.pt. Blog updated
(figs 9/10 re-rendered from 3k logs, both G1 clips replaced).

---

## G1 gait lab (2026-07-12): diagnosis + the stride fix

User verdict on the 3k gait: waddle/shuffle. Diagnostics on pod6:
- Stock policy filmed in ITS OWN 300 N.m world also crouch-walks at 0.75 m/s
  -> the crouch is the stock flat-task reward's signature at low speed, not
  (mainly) our actuator.
- Same 3k Atlas policy commanded 1.0 m/s: stride opens up substantially.

Arms:
- ext: resume 3k checkpoint +2000 iters (5000 total, stock recipe/gains).
  Reward 19.6 -> 19.9. Gait at 1.0 m/s: upright, striding, arm swing. Evals:
  unpushed 3.5% (vs 0.4% at 3k, noise-range), +/-1.0 push 21.3% (vs 41.7% —
  big robustness win). WINNER for footage.
- tuned: PD gains rescaled to torque ratio (legs 75/100, damping 3), fresh
  3000 iters. Reward 17.3. Gait visibly weak; evals 1.9% unpushed but 56.8%
  at +/-1.0. Gains-by-dimensional-analysis LOSES to optimizer-through-the-
  real-model. Kept as a negative result in the post.

Page: three-panel row (stock-in-fantasy / stock-on-real / Atlas-on-real) +
gait paragraph; Fig 10 and shove clip remain the 3k same-budget A/B.
Checkpoints: g1_gaitlab/film/policyB_g1_{ext,tuned}.pt.

---

## G1 upright gait (2026-07-12): posture reward + warm start

Stock reward is satisfied by a crouch (prices velocity, not posture), so no
training budget changes the gait family. Upright variants (Isaac-Velocity-
Flat-G1-Upright-v0 / -AtlasFOC-Upright-v0) add base_height_l2 (target 0.78,
w -100), flat_orientation -2.0, feet_air_time 1.0/thr 0.5 — identical in
both worlds. All A/B numbers of record remain stock-reward.

From scratch, 5000 iters each:
- Fantasy-torque world: walks upright immediately. (Reference clip.)
- Real physics: STUCK STANDING (0.4% unpushed falls, 97.8% at ±1.0 —
  stable statue). The height penalty blocks the crouch stepping-stone and
  real torque limits make walking hard to discover: fantasy torque changes
  which behaviors are REACHABLE, not just their quality.

Fix: warm-start from the walking crouch policy (policyB_g1_ext.pt) under
the upright reward, +2000 iters. Result: upright striding walk on real
physics, 0.0% falls unpushed (0/256), 32.0% at ±1.0 (crouch gait: 21.3%).
The upright gait trades some shove robustness for looking shippable; both
are one training flag apart in the same plugin.

Failure notes: pod uoajzvk2jlwoyx zombie (RUNNING, unreachable, 2.5h);
smoke-test prints need flush=True (Kit app.close() eats buffered stdout).
Checkpoints: g1_upright*/film/policy*_upright*.pt. Page panels now fully
matched: same posture reward, 1.0 m/s commands.

---

## Upright shove footage + co-design section (2026-07-13)

Refilmed the G1 shove comparison with the upright (posture-reward) pair so
every humanoid clip on the page walks upright: stock-upright vs Atlas-
upright-ft in the FOC world, 1.0 m/s shoves, vx 0.75, 15 s: 152 falls vs 4.
(First attempt died on a transient Isaac crash before the video wrote; the
retry pipeline no longer fail-fasts between films and always pulls logs.)
The page's reference panel now uses the 0.75 m/s stock-world clip (the
1.0 m/s one showed a leg-up hold that reads as broken; free torque lets the
fantasy policy hold poses no real robot would).

Also added the co-design section to the post: hardware design was open-loop
w.r.t. behavior; physics-from-design gives actuators a compile-test loop
(gear sweep by policies, optimizer-beats-dimensional-analysis on gains,
reward-actuator reachability), before the first part is machined.

---

## G1 upright v2 (2026-07-13): corrected reward, exploit removed, trade gone

The leg-hold tic was reward farming: amplifying feet_air_time (1.0/0.5) let
the fantasy-torque policy hold a leg mid-air for the bonus; the real-physics
policy could not afford the trick. Reverted stride term to stock (0.75/0.4),
posture terms unchanged, retrained both worlds (A from scratch 3000; B
warm-start from policyB_g1_ext.pt +1500).

Results (corrected pair, Atlas FOC world, 256 envs, 30 s):
- B: 0.0% falls unpushed (0/256), 22.4% at ±1.0 — statistically level with
  the crouch gait's 21.3. The upright gait no longer costs robustness.
- A in FOC: 100% (4554/4554), unchanged as ever.
- Shove films at 1.0: A 147 falls vs B 1 in 15 s.
- A-in-own-world footage: upright continuous walking, no leg-hold (verified
  across three windows + frame-diff motion check).

Ops notes: pod13 died at balance $0 mid-filming (checkpoints lost except
pre-pulled A; always pre-pull checkpoints); pod14 finished everything for
~$1.3 after top-up. Page panels now one recipe throughout; gait paragraph
documents the exploit as a reward-hygiene lesson.
Checkpoints: g1_up2/policy{A,B}_g1_up2.pt (+ film/ copies).

---

## Push curriculum + hardware effects (2026-07-15)

Two new tiers, both warm-started from the corrected upright walker
(policyB_g1_up2.pt), 1500 iters each:

- **Upright-Hard** (push curriculum ±1.0 in training): 0.0% falls unpushed,
  0.4% at ±1.0 (1 fall / 512 episodes). Curriculum closes the humanoid
  robustness gap entirely.
- **Upright-HW** (14-bit motor-shaft encoder + 8 mrad joint-side backlash):
  14.3% unpushed, 45.0% at ±1.0 — real, non-fatal degradation. v1 of the
  backlash model gated torque to zero for a full 5 ms step per reversal
  (~10x physical dead time) and collapsed the policy (92.5% unpushed);
  caught on film, fixed to time-resolved traversal at the motor's free
  speed (commit 507fb46). Only the corrected numbers are reported.

sense_encoder_backlash unit-tested (13/13). Checkpoints:
g1_hw/film/policyB_g1_up_{hard,hw}.pt. Page: ledger +2 rows, limitations
trimmed, comparison-table backlash row flipped.

---

## Multi-seed replication (2026-07-31): the 2.6x was the kindest draw

Seeds 43 and 44 retrained from scratch on pod A (1500 iters each policy,
same configs as v2), evals ~260 episodes/cell, 1-3 repeats:

| seed | A own-world | A on FOC (transfer) | B on FOC (matched) |
|---|---|---|---|
| 42 | 26.14 · 1.9% | 24.49 · 5.0% | 25.89 · 2.7% |
| 43 | 26.34 · 1.0% | 24.10 · 6.3% | 26.46 · 0.9% |
| 44 | 25.96 · 1.4% | 23.78 · 5.8% | 25.76 · 1.5% |
| **mean** | **26.15 · 1.4%** | **24.12 · 5.7%** | **26.04 · 1.7%** |

- Transfer penalty (A-foc vs A-own falls): 2.6x / 6.3x / 4.1x by seed, ~4x
  pooled. Seed 42 — the one the report originally led with — was the most
  flattering of the three.
- Parity replicates: B-foc within noise of A-own on every seed (B even beats
  the baseline on 43).
- s44 got fewer eval repeats (driver pulled while later repeats queued) and
  one s44 eval used model_1200 vs 1499; numbers are consistent across both.
- Raw: ~/Downloads/atlas_ab_experiment/seeds_quad/results.txt. Report edits:
  key-findings bullets, replication paragraph after Fig 5, limitations.

---

## Thermal-aware policy (2026-07-31): load-balancing, not pacing

Four attempts to make heat matter, each failure diagnostic:

1. **v1 (from scratch)**: standing trap again — both arms learned statues
   (vx 0.00). Thermal tasks need the warm start like the upright G1 did.
2. **v2 (warm start, R=1.8 K/W, derate 120 C)**: walks 1.7 m/s but peaks
   49 C — physics too gentle, nothing thermal happens. ThermalObs arm
   crashed in 16 s on obs-dim mismatch (48→60) and Kit exited 0; caught by
   pipeline timestamps, fixed by checkpoint surgery (zero-pad actor/critic
   first layers + Adam state so both arms start byte-identical).
3. **v3 (R=5.0, derate 85 C)**: peaks 75.5 C at 60 s — tau = R*C = 40 s
   means a 60 s episode ends at the derate line's doorstep.
4. **v4 (120 s episodes)**: constraint binds. Eval gotcha: eval_thermal.py
   defaulted --steps 3000 (60 s) and measured only the cool half; rerun
   with --steps 6000 on a fresh pod.

Result (4096 envs, 120 s, commands 1.0-2.5 m/s, tag = fleet-hottest winding):

| arm | mean vx | peak T_max | T_max >= 80 C from | falls |
|---|---|---|---|---|
| blind | 1.68 m/s | 91.9 C | t = 30 s | 17 |
| temp-aware | 1.71 m/s | 82.1 C | t = 90 s | 4 |

The aware policy does NOT slow down (naive prediction): same speed, hottest
winding ~11 C cooler at t=90 — it redistributes effort so no single winding
becomes the fuse. Thermal load-balancing appears in no reward term; it is
what the optimizer does with a temperature sense + physics where heat costs.
Figure: figs/fig_thermal.png (script ~/.atlas_ops/figscripts/fig_thermal.py),
report Fig 12. Checkpoints: ~/Downloads/atlas_ab_experiment/thermal_v4/film/;
evals: thermal_v4/eval120/results.txt. Both arms warm-started from
policy_th2_Thermal.pt (obs arm via the padded twin, so t=0 behavior identical).

---

## G1 multi-seed (2026-07-31): replicated at 25x; seed 44 lost to silent crash

Seed 43 retrained both G1 upright policies from scratch on pod B (3000
iters each), evaluated on real FOC physics, no pushes, two repeats:

| seed | stock recipe on FOC | Atlas-trained on FOC |
|---|---|---|
| 42 | 100% falls (4,554/4,554) | 0.0% (0/256) |
| 43 | 99.8% / 99.6% (4,563/4,574) | 4.2% / 4.9% (11/260) |

The catastrophic-vs-walks split replicates. Seed 44: A-arm trained fully
but the B-arm died silently twice (log ends mid-iteration, no error, Kit
exit-code swallowing again — second death ~575/3000 iters), and the
driver's poll budget expired and terminated the pod before s44 artifacts
could be pulled. Not rerun: two seeds at a 25x effect size settle the
claim; report limitations updated to say "three seeds quadruped, two
humanoid" honestly. Raw: ~/Downloads/atlas_ab_experiment/seeds_g1/.
