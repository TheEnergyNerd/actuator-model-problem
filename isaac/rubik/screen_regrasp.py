"""Check native hand hull overlap along U-to-F regrasp candidates."""

import json
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
import collision_geometry as g
from regrasp import FrontRegrasp

hand = g.models["right"][0]
sr = np.array(g.cfg["Sr"])
old_frame = g.R @ Rotation.from_euler("z", -90, degrees=True).as_matrix() @ sr
center = np.array([0, 0, 0.5])
old_center = center + 0.01905 * (g.R[:, 2] - old_frame[:, 2])
q = Rotation.from_matrix(g.R).as_quat()
pose = np.r_[center, q[3], q[:3]]
results = []
for face, roll in __import__("itertools").product(
    ["B", "R", "L", "D"], [0, 90, 180, -90]
):
    controller = FrontRegrasp(
        SimpleNamespace(hand=hand, q=g.qs["right"]),
        g.R,
        sr,
        old_center,
        0.009,
        roll,
        face,
    )
    overlaps = []
    for t in np.arange(8.5, 18.501, 0.5):
        frame, cp, joints, phase = controller.command(t, pose)
        g.poses["right"] = hand.wrist_pose(cp, frame)
        g.right = g.shapes_world("right", joints)
        hits = g.collisions(g.qs["left"])
        if hits:
            overlaps.append(dict(t=round(float(t), 2), phase=phase, hits=hits))
    results.append(
        dict(
            face=face,
            roll=roll,
            max_overlap_m=max([max(r["hits"].values()) for r in overlaps] + [0]),
            overlaps=overlaps,
        )
    )
print(json.dumps(results, indent=2))
