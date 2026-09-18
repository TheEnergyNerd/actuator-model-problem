"""Run a pinned external pen checkpoint with Atlas recording hooks.

Reference code is fetched separately, not vendored. The baseline physics and
policy observations are unchanged; optional interventions are recorded explicitly. Rendering/scoring use the recorded Isaac states, with no MuJoCo.
"""

import argparse
import hashlib
import math
from pathlib import Path
import sys

REVISION = "a1c93b3ab814995b0c75bae51595417caa91c34b"
CHECKPOINT = "cc88085343402380b07df560a874b7784e05ed9fb8f75e0b19c575c7e08bf4cc"


def prepare(
    reference: Path,
    output: Path,
    effort_scale=1.0,
    mass_scale=1.0,
    model="native",
    policy_checkpoint: Path | None = None,
):
    import json

    if not all(math.isfinite(v) and 0 < v <= 4 for v in (effort_scale, mass_scale)):
        raise ValueError("Hardware scales must be finite and between zero and four")

    hashes = json.loads(Path(__file__).with_name("reference-files.json").read_text())
    for relative, expected in hashes.items():
        if hashlib.sha256((reference / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reference file mismatch: {relative}")
    pen = reference / "pen"
    checkpoint = pen / "checkpoints/best_policy.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT:
        raise ValueError("Reference checkpoint hash mismatch")
    evaluated_checkpoint = policy_checkpoint or checkpoint
    evaluated_hash = hashlib.sha256(evaluated_checkpoint.read_bytes()).hexdigest()
    source = (pen / "policy/evaluate.py").read_text()
    replacements = {
        "weights_only=False": "weights_only=True",
        "steps = int(args.seconds / CONTROL_DT)": f"from intervention import apply_intervention\napply_intervention(env, args.out, {effort_scale!r}, {mass_scale!r})\nfrom motor_model import attach\nattach(env, args.out, {model!r})\nfrom recording import PenRecording\nrecorder = PenRecording(args.out, env)\nsteps = int(args.seconds / CONTROL_DT)",
        "        if (term | trunc).any():": "        recorder.capture()\n        if (term | trunc).any():",
        "app.close()": "recorder.finish()\nimport threading, os\n_shutdown = threading.Timer(10, lambda: os._exit(0))\n_shutdown.daemon = True\n_shutdown.start()\napp.close()",
    }
    for before, after in replacements.items():
        if before not in source:
            raise ValueError(f"Reference integration point changed: {before}")
        source = source.replace(before, after)
    # Preserve external module/resource paths when evaluating the instrumented source.
    sys.path.insert(0, str(pen / "sim"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "reference.json").write_text(
        __import__("json").dumps(
            {
                "repository": "https://github.com/jianglongye/dexterous-astra",
                "revision": REVISION,
                "checkpoint_sha256": evaluated_hash,
                "initial_reference_checkpoint_sha256": CHECKPOINT,
                "policy_origin": (
                    "Atlas fine-tuned checkpoint"
                    if policy_checkpoint
                    else "Supplied reference policy"
                ),
                "changes": [
                    "weights-only checkpoint loading",
                    "measured body and torque recording",
                ],
                "selected_trial": 0,
                "effort_scale": effort_scale,
                "hand_mass_scale": mass_scale,
                "motor_model": model,
                "selection": "fixed before evaluation",
                "penetration_method": "PhysX contact separation; different from upstream MuJoCo hull check",
            },
            indent=2,
        )
    )
    return source, pen


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--trials", type=int, default=32)
    p.add_argument("--seed0", type=int, default=41000000)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--effort-scale", type=float, default=1.0)
    p.add_argument("--hand-mass-scale", type=float, default=1.0)
    p.add_argument(
        "--model", choices=["native", "ideal", "motor", "motor-hot"], default="native"
    )
    args, rest = p.parse_known_args()
    source, pen = prepare(
        args.reference.resolve(),
        args.out.resolve(),
        args.effort_scale,
        args.hand_mass_scale,
        args.model,
        args.checkpoint.resolve() if args.checkpoint else None,
    )
    sys.argv = [
        str(pen / "policy/evaluate.py"),
        "--ckpt",
        str(
            args.checkpoint.resolve()
            if args.checkpoint
            else pen / "checkpoints/best_policy.pt"
        ),
        "--run_cfg",
        str(pen / "checkpoints/env_cfg.json"),
        "--out",
        str(args.out.resolve()),
        "--trials",
        str(args.trials),
        "--seed0",
        str(args.seed0),
        "--seconds",
        "12",
        *rest,
    ]
    exec(
        compile(source, str(pen / "policy/evaluate.py"), "exec"),
        {"__name__": "__main__", "__file__": str(pen / "policy/evaluate.py")},
    )
