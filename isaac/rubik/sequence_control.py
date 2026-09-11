"""Finite-state wrist/grasp commands. No cube pose access or cube actuation."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from cube_state import NORMALS, FACES


def smooth(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def frame_for(face):
    n = NORMALS[FACES.index(face)]
    z = np.array([0, 0, 1])
    if n[2] == -1:
        return Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
    axis = np.cross(z, n)
    return (
        np.eye(3)
        if np.linalg.norm(axis) < 0.1
        else Rotation.from_rotvec(axis * np.pi / 2).as_matrix()
    )


class Sequence:
    period = 13.0

    def __init__(self, hand, moves, closure=0.004):
        self.hand = hand
        self.moves = [
            q for m in moves.split() for q in ([m[0], m[0]] if m.endswith("2") else [m])
        ]
        self.frames = []
        self.entry_frames = {}
        repeated = 0
        for i, m in enumerate(self.moves):
            repeated = repeated + 1 if i and m[0] == self.moves[i - 1][0] else 0
            self.frames.append(
                frame_for(m[0])
                @ Rotation.from_rotvec(
                    [
                        0,
                        0,
                        ((repeated % 2) + {"R": 1, "B": 2, "L": -1}.get(m[0], 0))
                        * np.pi
                        / 2,
                    ]
                ).as_matrix()
            )
        self.angles = [
            -np.pi / 2 * (2 if m.endswith("2") else -1 if m.endswith("'") else 1)
            for m in self.moves
        ]
        self.open = hand.grasp(-0.012)[0]
        self.closed = hand.grasp(closure)[0]

    def command(self, t):
        index = min(int(t / self.period), len(self.moves) - 1)
        s = t - index * self.period
        base = self.frames[index]
        angle = self.angles[index]
        stretch = 1.0
        closed = 0.0
        phase = "Approach"
        if s < 2:
            if index or index in self.entry_frames:
                previous = self.entry_frames.get(
                    index,
                    (
                        self.frames[index - 1]
                        @ Rotation.from_rotvec(
                            [0, 0, self.angles[index - 1]]
                        ).as_matrix()
                    ),
                )
                base = Slerp([0, 1], Rotation.from_matrix([previous, base]))(
                    smooth(s / 2)
                ).as_matrix()
            stretch = 2.3
            phase = "Reposition"
        elif s < 3.5:
            stretch = 2.3 - 1.3 * smooth((s - 2) / 1.5)
        elif s < 4.3:
            closed = smooth((s - 3.5) / 0.8)
            phase = "Grasp"
        elif s < 8.3:
            closed = 1.0
            base = (
                base
                @ Rotation.from_rotvec(
                    [0, 0, angle * smooth((s - 4.3) / 4)]
                ).as_matrix()
            )
            phase = "Turn"
        else:
            base = base @ Rotation.from_rotvec([0, 0, angle]).as_matrix()
            if s < 9.5:
                closed = 1.0
                phase = "Verify"
            elif s < 10.3:
                closed = 1 - smooth((s - 9.5) / 0.8)
                phase = "Release"
            else:
                stretch = 1 + 1.3 * smooth((s - 10.3) / 1.7)
                phase = "Retract" if s < 12 else "Settle"
        pos, rot = self.hand.wrist_pose([0, 0, 0.5], base)
        pos = np.array([0, 0, 0.5]) + stretch * (pos - np.array([0, 0, 0.5]))
        joints = {
            n: self.open[n] + closed * (self.closed[n] - self.open[n])
            for n in self.open
        }
        return pos, rot, joints, index, phase
