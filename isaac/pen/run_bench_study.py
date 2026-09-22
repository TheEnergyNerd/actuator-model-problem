"""Reproduce the three-seed bench-informed study using the Isaac Python runtime.

Set SHARPA_ROOT to the separately fetched, pinned official mesh assets. Run:
/isaac-sim/python.sh run_bench_study.py --reference /path/to/dexterous-astra --out /new/output
This serial runner prioritizes reproducibility; independent training seeds can
also run on separate GPUs. It does not create cloud resources or contact hardware.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

SEEDS = (44000000, 44010000, 44020000)
MODELS = ("ideal", "motor", "motor-hot")
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "ATLAS_MOTOR_PROFILE": "mj5208-virtual"}
    reference = str(args.reference.resolve())

    def run(script, *flags):
        subprocess.run(
            [sys.executable, str(HERE / script), *map(str, flags)], env=env, check=True
        )

    def evaluate(path, model, seed, trials, checkpoint=None):
        flags = [
            "--reference",
            reference,
            "--out",
            path,
            "--model",
            model,
            "--seed0",
            seed,
            "--trials",
            trials,
            "--headless",
            "--device",
            args.device,
        ]
        if checkpoint:
            flags += ["--checkpoint", checkpoint]
        run("run_reference.py", *flags)
        run("export_run.py", path, "--reference", reference)

    for seed in SEEDS:
        for model in ("ideal", "motor"):
            run(
                "train_matched.py",
                "--reference",
                reference,
                "--out",
                root / "training" / f"{seed}-{model}",
                "--model",
                model,
                "--num-envs",
                256,
                "--iterations",
                256,
                "--rollout-steps",
                64,
                "--train-seed",
                seed,
                "--temperature-min",
                25,
                "--temperature-max",
                25,
                "--test-seed0",
                46000000,
                "--test-trials",
                128,
                "--headless",
                "--device",
                args.device,
            )
    for seed in SEEDS:
        for model in ("ideal", "motor"):
            for iteration in (64, 128, 256):
                evaluate(
                    root / "validation" / f"{seed}-{model}-{iteration:04d}",
                    "motor",
                    45000000,
                    32,
                    root
                    / "training"
                    / f"{seed}-{model}"
                    / f"policy-{iteration:04d}.pt",
                )
    for model in MODELS:
        evaluate(root / "evaluation" / f"frozen-{model}", model, 46000000, 128)
        for seed in SEEDS:
            for policy in ("ideal", "motor"):
                evaluate(
                    root / "evaluation" / f"{seed}-{policy}-{model}",
                    model,
                    46000000,
                    128,
                    root / "training" / f"{seed}-{policy}" / "policy-0256.pt",
                )
    (root / "COMPLETE").touch()


if __name__ == "__main__":
    main()
