"""Native hand collision hulls for conservative kinematic screening, not physical validation."""

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
