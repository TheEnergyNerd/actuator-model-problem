# Co-design sweep: choosing gear ratio by learned task performance

**Date:** 2026-07-09 · **Hardware:** RunPod L40S · **Cost:** ~$0.53 ·
**Method:** same motor design, three gear ratios (6:1 / 9:1 / 12:1) as three
Isaac Lab tasks; PPO per variant (4096 envs, 300 iters, single seed); each
policy evaluated in its own world (256 envs, 1500 steps, ~260 episodes).

## Results — ANYmal-C flat-ground velocity tracking

| gear | peak joint torque | joint base speed | eval mean ep reward | fall rate | final train reward |
|---|---|---|---|---|---|
| 6:1  | 74 N·m  | ~21 rad/s | 3.28 | 1.2% | 1.76 |
| 9:1  | 111 N·m | ~14 rad/s | **3.36** | 3.4% | 0.13 |
| 12:1 | 148 N·m | ~11 rad/s | 3.26 | 0.8% | −0.11 |

## Finding (honest)

Eval performance is **statistically flat** across the sweep (Δ ≤ 0.10 with a
single seed; fall counts 2–9 of ~260). The hypothesized interior optimum did
not appear: flat-ground walking at these commanded velocities does not stress
the actuator anywhere in the 74–148 N·m range.

**The co-design verdict is still actionable:** choose the 6:1 — the smallest,
lightest, cheapest gearbox with the fastest joints — because the sweep shows
the extra torque of 9:1/12:1 buys nothing *for this task*. Knowing which
design axis you don't need to pay for is half of what co-design is for, and
this answer was measured (three trained policies), not guessed from datasheets.

Secondary signal: final *training* rewards ordered 6:1 > 9:1 > 12:1 (1.76 /
0.13 / −0.11) — faster joints appear easier to learn with, though train-time
reward includes exploration noise and is weaker evidence than the eval.

## What would make the axis discriminate

The task, not the sweep, is the limiting factor. Rough terrain, sprint-speed
commands, payload, or stairs demand torque-at-speed where the gear tradeoff
bites (6:1's 74 N·m vs 12:1's 11 rad/s ceiling). The pipeline is push-button:
register the harder task, rerun the same driver (~30 min, <$1).

## Why this matters

This experiment is structurally impossible in the standard workflow: it
compares *learned* task performance across actuator designs **before any
hardware exists**. The standard path would require building three gearboxes
(or trusting datasheet intuition). Total cost here: 53 cents and ~35 minutes.

Artifacts: `sweep_g6/g9/g12.mp4` (policies walking in their own worlds),
`sweep_raw.txt` (eval JSON + train rewards), driver `/opt/run_sweep.sh`
pattern in the A/B writeup.

---

## Sweep v2 (2026-07-13): corrected gains, motion-verified footage

Retrained all three gear policies (1500 iters, Kp 85/Kd 2) after the user
caught the old montage showing near-static robots (raw omnidirectional
commands sample near-zero velocities). New evals, own world, ~260 episodes:

| gear | peak N·m | mean ep reward | falls |
|---|---|---|---|
| 6:1 | 74 | 25.04 | 4.2% |
| 9:1 | 111 | 25.88 | 2.7% |
| 12:1 | 148 | 26.08 | 2.7% |

Near-parity holds (9:1/12:1 inside the Fig 4 baseline band 25.9-26.1, so the
old "predates the gain fix" caveat is gone). New fine structure: the 6:1
pays ~1 reward point and +1.5pp falls — a small recovery-headroom tax on the
lightest gearbox. Montage refilmed at fixed 1.0 m/s forward, motion-verified
(~6.2 mean px diff vs ~3 for the dead clip). Checkpoints:
~/Downloads/atlas_ab_experiment/sweep2/film/policy_sweep2_g{6,9,12}.pt.

---

## Sprint sweep (2026-07-30): the axis that discriminates

Sweep v2's conclusion was "flat cruise can't rank gearboxes; change the task
to make the axis discriminate." Done: `*-Sprint-v0` variants command
lin_vel_x in (0, 2.5) m/s (vs 1.0 fixed), everything else identical.
1500 iters per gear on pod C, two eval repeats each (~280 episodes/repeat):

| gear | mean ep reward | falls |
|---|---|---|
| 6:1  | 10.07 / 10.38 | 16.7% / 17.3% |
| 9:1  | 7.53 / 7.88   | 19.5% / 18.8% |
| 12:1 | 8.29 / 8.38   | 12.3% / 12.2% |

The tie breaks: 6:1 fastest but falls most, 12:1 most robust, and 9:1 is
**strictly dominated by 12:1** (less reward AND more falls) — the compromise
gear is the only indefensible pick. Reward scale is not comparable to the
cruise table (different command distribution). Figure:
figs/fig_sprint_pareto.png (script ~/.atlas_ops/figscripts/fig_sprint_pareto.py);
report Fig 11. Checkpoints/logs: ~/Downloads/atlas_ab_experiment/sprint_thermal/.
