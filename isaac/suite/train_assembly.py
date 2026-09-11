"""Bounded sequential native Factory training, evaluation and measured recording."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from snapshot_checkpoint import snapshot


def run(command, log, timeout, env, cwd):
    with log.open("w") as out:
        process = subprocess.Popen(
            command,
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--isaaclab", type=Path, required=True)
    p.add_argument("--deps", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wait-pid", type=int)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    # Wait only for this session's peg trainer, never an unrelated process.
    while a.wait_pid:
        proc = Path(f"/proc/{a.wait_pid}/cmdline")
        if not proc.exists():
            break
        command = proc.read_bytes()
        if b"atlas_suite_peg_20260911" not in command:
            break
        time.sleep(30)
    env = dict(
        os.environ,
        PYTHONPATH=str(a.deps),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    python = [str(a.isaaclab / "isaaclab.sh"), "-p"]
    summaries = []
    for task in ["GearMesh", "NutThread"]:
        name = f"atlas_suite_{task.lower()}_{int(time.time())}"
        out = a.output / task
        out.mkdir()
        train = python + [
            "scripts/reinforcement_learning/rl_games/train.py",
            "--headless",
            "--device",
            "cuda:0",
            "--task",
            f"Isaac-Factory-{task}-Direct-v0",
            "--num_envs",
            "128",
            "--seed",
            "42",
            "--max_iterations",
            "200",
            f"agent.params.config.full_experiment_name={name}",
        ]
        (out / "command.json").write_text(json.dumps(train, indent=2))
        status = run(train, out / "training.log", 7200, env, a.isaaclab)
        result = dict(task=task, training_exit=status, experiment=name)
        checkpoint = a.isaaclab / "logs/rl_games/Factory" / name / "nn/Factory.pth"
        if checkpoint.exists():
            try:
                snapshot(checkpoint, out / "checkpoint.pt")
                result["evaluation_exit"] = run(
                    python
                    + [
                        str(source / "eval_factory.py"),
                        "--headless",
                        "--device",
                        "cuda:0",
                        "--task",
                        task,
                        "--record",
                        "--checkpoint",
                        str(out / "checkpoint.pt"),
                        "--output",
                        str(out / "evaluation"),
                    ],
                    out / "evaluation.log",
                    900,
                    env,
                    a.isaaclab,
                )
                report = out / "evaluation/result.json"
                if report.exists():
                    result["evaluation"] = json.loads(report.read_text())
                # Render retained first episode whether it succeeds or fails.
                if (out / "evaluation/replay/result.json").exists():
                    result["render_exit"] = run(
                        python
                        + [
                            str(source / "render_recording.py"),
                            "--headless",
                            "--device",
                            "cpu",
                            "--fps",
                            "15",
                            "--recording",
                            str(out / "evaluation"),
                        ],
                        out / "render.log",
                        900,
                        env,
                        a.isaaclab,
                    )
            except Exception as error:
                result["error"] = str(error)
        summaries.append(result)
        (a.output / "results.json").write_text(json.dumps(summaries, indent=2) + "\n")
