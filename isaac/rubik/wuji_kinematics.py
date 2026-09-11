"""URDF forward kinematics and bounded fingertip IK. No simulator dependency."""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

# Numerical grasp seeds published with the reference demo; see README attribution.
SEEDS = {
    "left": [
        0.515660,
        -0.954237,
        1.059029,
        -0.094306,
        0.241842,
        0.544055,
        1.346823,
        -0.072454,
        0.014070,
        -0.070162,
        -0.088172,
        -0.965351,
        -0.065736,
        0.586783,
    ],
    "right": [
        0.561351,
        -1.150161,
        0.949888,
        0.248666,
        0.169256,
        0.403329,
        1.843982,
        -0.594934,
        0.017517,
        0.069728,
        -0.080241,
        1.031435,
        0.584936,
        0.053142,
    ],
}


class Hand:
    def __init__(self, side, layer=0.01905):
        self.layer = layer
        self.side = side
        self.prefix = side[0] + "_"
        root = ET.parse(
            Path(__file__).parent
            / f"assets/wuji/hand2/hand2_beta1/body/urdf/{side}.urdf"
        ).getroot()
        self.joints = []
        for j in root.findall("joint"):
            o = j.find("origin")
            t = np.eye(4)
            t[:3, 3] = np.fromstring(o.get("xyz", "0 0 0"), sep=" ")
            t[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(o.get("rpy", "0 0 0"), sep=" ")
            ).as_matrix()
            a = j.find("axis")
            lim = j.find("limit")
            self.joints.append(
                (
                    j.get("name"),
                    j.find("parent").get("link"),
                    j.find("child").get("link"),
                    t,
                    (
                        None
                        if j.get("type") == "fixed"
                        else np.fromstring(a.get("xyz"), sep=" ")
                    ),
                    (
                        None
                        if lim is None
                        else (float(lim.get("lower")), float(lim.get("upper")))
                    ),
                )
            )
        self.names = [j[0] for j in self.joints if j[4] is not None]
        self.seed = np.array(SEEDS[side])
        self.Q = Rotation.from_rotvec(self.seed[11:14]).as_matrix()
        self.C = self.seed[8:11] - self.Q @ np.array([0, 0, self.layer])
        self.active = self.names[:8]

    def fk(self, q):
        transforms = {self.prefix + "wrist": np.eye(4)}
        for name, parent, child, t, axis, _ in self.joints:
            motion = np.eye(4)
            if axis is not None:
                motion[:3, :3] = Rotation.from_rotvec(axis * q.get(name, 0)).as_matrix()
            transforms[child] = transforms[parent] @ t @ motion
        return transforms

    def tips(self, values):
        frames = self.fk(dict(zip(self.active, values)))
        offset = np.array([0, 0.005 if self.side == "right" else -0.005, -0.023, 1])
        return np.array(
            [
                (frames[self.prefix + n] @ offset)[:3]
                for n in ("thumb_distal", "index_finger_distal")
            ]
        )

    def grasp(self, closure=0.003):
        sign = 1 if self.side == "right" else -1
        targets = np.array(
            [
                self.C + self.Q @ np.array([0, sign * (0.0345 - closure), self.layer]),
                self.C + self.Q @ np.array([0, -sign * (0.031 - closure), self.layer]),
            ]
        )
        limits = np.array([j[5] for j in self.joints if j[0] in self.active])
        result = least_squares(
            lambda q: np.r_[
                100 * (self.tips(q) - targets).ravel(), 0.002 * (q - self.seed[:8])
            ],
            np.clip(self.seed[:8], limits[:, 0], limits[:, 1]),
            bounds=(limits[:, 0], limits[:, 1]),
        )
        q = dict(zip(self.active, map(float, result.x)))
        return q, float(np.max(np.linalg.norm(self.tips(result.x) - targets, axis=1)))

    def wrist_pose(self, cube_pos, cube_rotation=np.eye(3)):
        w = cube_rotation @ self.Q.T
        return np.asarray(cube_pos) - w @ self.C, w

    def support_finger(self, finger, cube_point, frame, seed=None):
        names = [n for n in self.names if n.startswith(self.prefix + finger + "_")]
        limits = np.array([j[5] for j in self.joints if j[0] in names])
        seed = np.clip(
            np.array([0.0, 0.0, 1.0, 0.3]) if seed is None else seed,
            limits[:, 0],
            limits[:, 1],
        )
        wrist_pos, wrist_rotation = self.wrist_pose([0, 0, 0.5], frame)
        target = wrist_rotation.T @ (
            np.array([0, 0, 0.5]) + np.asarray(cube_point) - wrist_pos
        )
        offset = np.array([0, 0.005 if self.side == "right" else -0.005, -0.023, 1])

        def point(q):
            return (
                self.fk(dict(zip(names, q)))[self.prefix + finger + "_distal"] @ offset
            )[:3]

        result = least_squares(
            lambda q: np.r_[100 * (point(q) - target), 0.001 * (q - seed)],
            seed,
            bounds=(limits[:, 0], limits[:, 1]),
        )
        return dict(zip(names, map(float, result.x))), float(
            np.linalg.norm(point(result.x) - target)
        )
