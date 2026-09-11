"""Conservative static self/inter-hand collision screen for support-finger IK.

Convex hull screening is not grasp validation. This checks prepared end poses;
a collision-free end pose does not establish a safe approach or stable contact.
"""

import argparse
import sys, json, itertools
import numpy as np
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from wuji_kinematics import Hand

root = Path(__file__).parent
cfg = json.load(open(root / "reference_grasp.json"))
R = np.array(cfg["R"])
native = json.load(open(root / "native_collision_exclusions.json"))
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
a = parser.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
models = {}
qs = {}
poses = {}
ignored = set()
for side, layer in [("left", 0), ("right", 0.01905)]:
    h = Hand(side, layer)
    frame = R @ np.array(cfg["Sl" if side == "left" else "Sr"])
    pos, rot = h.wrist_pose(
        np.array([0, 0, 0.5]) + layer * (R[:, 2] - frame[:, 2]), frame
    )
    q = h.grasp(0.009)[0]
    if side == "left":
        for f in ["middle_finger", "ring_finger", "pinky"]:
            q["l_" + f + "_mcp_flex"] = -0.65
    qs[side] = q
    poses[side] = (pos, rot)
    zero = h.fk({})
    shapes = {}
    stage = Usd.Stage.Open(
        str(root / f"assets/wuji/hand2/hand2_beta1/body/usd/{side}/wujihand2.usd")
    )
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        name = str(prim.GetPath()).split("/")[2]
        m = np.array(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        )
        v = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get()) @ m[:3, :3] + m[3, :3]
        v = (v - zero[name][:3, 3]) @ zero[name][:3, :3]
        hull = ConvexHull(v)
        shapes[name] = (
            v[hull.vertices],
            np.unique(np.round(hull.equations, 9), axis=0),
        )
    models[side] = (h, shapes)
    for _, parent, child, *_ in h.joints:
        ignored.add(frozenset([parent, child]))
    for pair in native[side]:
        ignored.add(frozenset([pair["body1"], pair["body2"]]))


def shapes_world(side, q):
    h, shapes = models[side]
    fk = h.fk(q)
    pos, rot = poses[side]
    out = {}
    for name, (v, e) in shapes.items():
        b = fk[name]
        w = rot @ b[:3, :3]
        p = pos + rot @ b[:3, 3]
        points = v @ w.T + p
        normals = e[:, :3] @ w.T
        planes = np.c_[normals, e[:, 3] - normals @ p]
        out[name] = (points.min(0), points.max(0), planes)
    return out


right = shapes_world("right", qs["right"])


def collisions(q):
    shapes = shapes_world("left", q) | right
    overlaps = {}
    for an, bn in itertools.combinations(shapes, 2):
        if (
            an.startswith("r_")
            and bn.startswith("r_")
            or frozenset([an, bn]) in ignored
        ):
            continue
        al, ah, ap = shapes[an]
        bl, bh, bp = shapes[bn]
        if np.any(ah < bl) or np.any(bh < al):
            continue
        e = np.r_[ap, bp]
        res = linprog(
            [0, 0, 0, -1],
            A_ub=np.c_[e[:, :3], np.ones(len(e))],
            b_ub=-e[:, 3],
            bounds=[(None, None)] * 4,
            method="highs",
        )
        if res.success and res.x[3] > 0.00005:
            overlaps[an + " / " + bn] = float(res.x[3])
    return overlaps


base = collisions(qs["left"])
print("BASE", base, flush=True)
results = []
h = models["left"][0]
for finger in ["middle_finger", "ring_finger", "pinky"]:
    for y, z in itertools.product([-0.020, -0.010, 0], [-0.012, 0, 0.010]):
        target = np.array([-0.024, y, z])
        q, e = h.support_finger(finger, R @ target, R @ np.array(cfg["Sl"]))
        if e > 0.0002:
            continue
        hits = collisions(qs["left"] | q)
        score = max([v - base.get(k, 0) for k, v in hits.items()] + [0])
        row = dict(
            finger=finger,
            target=target.tolist(),
            ik_error_m=e,
            extra_overlap_m=score,
            collisions=hits,
            joints=q,
        )
        results.append(row)
results.sort(key=lambda x: x["extra_overlap_m"])
a.output.write_text(
    json.dumps(
        dict(
            method="Native collision-mesh convex hulls; prepared poses only",
            closure_m=0.009,
            cube_contacts_screened=False,
            baseline_collisions=base,
            candidates=results,
        ),
        indent=2,
    )
)
for row in results[:8]:
    print(json.dumps(row), flush=True)
