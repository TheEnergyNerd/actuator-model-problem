"""Check that the native fingers can reach a grasp in free space without self-collision jams."""

import argparse, json
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app
import numpy as np
import isaaclab.sim as sim_utils
from native_hand import MountedHand

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.001, device=a.device))
centers = [(-0.25, 0, 0.5), (0.25, 0, 0.5)]
hands = [
    MountedHand(sim.stage, side, 0.01905, np.eye(3), 0.003, cube_pos=pos)
    for side, pos in zip(("left", "right"), centers)
]
sim.reset()
for h in hands:
    h.initialize(sim.device)
for step in range(1000):
    for h, pos in zip(hands, centers):
        h.command(cube_pos=pos)
    sim.step(render=False)
    for h in hands:
        h.robot.update(0.001)
rows = []
for h in hands:
    error = float((h.robot.data.joint_pos - h.target).abs().max())
    torque = float(h.robot.data.applied_torque.abs().max())
    rows.append(
        {
            "hand": h.hand.side,
            "max_joint_error_rad": error,
            "max_torque_nm": torque,
            "passed": error < 0.01 and torque < 0.03,
            "contact_model": h.contact_model,
        }
    )
a.output.write_text(json.dumps(rows, indent=2))
print("RESULT", json.dumps(rows), flush=True)
import threading, os

threading.Timer(5, lambda: os._exit(0)).start()
app.close()
