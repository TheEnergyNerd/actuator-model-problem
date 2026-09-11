"""Run the suite's native environment checks sequentially in fresh Isaac processes."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

TASKS = {
    "anymal": "Isaac-Velocity-Rough-Anymal-C-Play-v0",
    "anymal-atlas": "Isaac-Velocity-Rough-Anymal-C-AtlasFOC-v0",
    "peg": "Isaac-Factory-PegInsert-Direct-v0",
    "gear": "Isaac-Factory-GearMesh-Direct-v0",
    "nut": "Isaac-Factory-NutThread-Direct-v0",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--timeout", type=float, default=360)
    parser.add_argument("--extension", type=Path)
    args = parser.parse_args()
    if args.timeout <= 0 or len(set(args.tasks)) != len(args.tasks):
        parser.error("timeout must be positive and tasks must be unique")
    if "anymal-atlas" in args.tasks and args.extension is None:
        parser.error("anymal-atlas requires --extension /path/to/atlas_actuators_ext")
    launcher = args.isaaclab.resolve() / "isaaclab.sh"
    if not launcher.is_file():
        parser.error("Isaac Lab launcher not found")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1"
    )
    if args.extension:
        environment["PYTHONPATH"] = (
            str(args.extension.resolve())
            + os.pathsep
            + environment.get("PYTHONPATH", "")
        )
    results = []
    for name in args.tasks:
        command = [
            str(launcher),
            "-p",
            str(Path(__file__).with_name("smoke.py").resolve()),
            "--headless",
            "--device",
            "cuda:0",
            "--task",
            TASKS[name],
            "--output",
            str(args.output / name),
        ]
        if name == "anymal-atlas":
            command.append("--atlas")
        record = {
            "name": name,
            "task": TASKS[name],
            "passed": False,
            "command": command,
        }
        print(f"Starting {name}", flush=True)
        with (args.output / f"{name}.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=args.isaaclab,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                record["exit_code"] = process.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                record["error"] = type(error).__name__
                record["exit_code"] = process.returncode
                if isinstance(error, KeyboardInterrupt):
                    raise
        path = args.output / name / "result.json"
        if path.exists():
            try:
                record["result"] = json.loads(path.read_text())
                record["passed"] = (
                    record["exit_code"] == 0
                    and "error" not in record
                    and record["result"].get("passed") is True
                    and record["result"].get("task") == TASKS[name]
                )
            except (ValueError, TypeError):
                record["error"] = "Invalid result JSON"
        results.append(record)
        (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"{name}: {'PASS' if record['passed'] else 'FAIL'}", flush=True)
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
