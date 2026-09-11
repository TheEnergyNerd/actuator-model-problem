"""World-frame U/F wrist path with a physical release and regrasp between turns."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def smooth(value):
    x = np.clip(value, 0, 1)
    return x * x * (3 - 2 * x)


class FrontRegrasp:
    def __init__(self, hand, reference, sr, old_center, closure, roll=0, face="F"):
        self.hand = hand
        self.reference = reference
        self.sr = sr
        self.old_center = old_center
        self.closed = hand.q | hand.hand.grasp(closure)[0]
        self.open = hand.q | hand.hand.grasp(-0.008)[0]
        self.old_frame = (
            reference @ Rotation.from_euler("z", -90, degrees=True).as_matrix() @ sr
        )
        self.target_frame = None
        self.roll = roll
        self.face = face

    def command(self, t, core_pose):
        if self.target_frame is None:
            self.center = np.asarray(core_pose[:3]).copy()
            self.core_frame = Rotation.from_quat(core_pose[[4, 5, 6, 3]]).as_matrix()
            axis, degrees = {
                "F": ("x", 90),
                "B": ("x", -90),
                "R": ("y", 90),
                "L": ("y", -90),
                "D": ("x", 180),
            }[self.face]
            self.face_frame = (
                self.core_frame
                @ Rotation.from_euler(axis, degrees, degrees=True).as_matrix()
                @ Rotation.from_euler("z", self.roll, degrees=True).as_matrix()
            )
            self.target_frame = self.face_frame @ self.sr
            self.target_center = self.center + 0.01905 * (
                self.face_frame[:, 2] - self.target_frame[:, 2]
            )
            self.old_out = (
                self.hand.hand.wrist_pose(self.old_center, self.old_frame)[0]
                - self.center
            )
            self.old_out /= np.linalg.norm(self.old_out)
            self.new_out = (
                self.hand.hand.wrist_pose(self.target_center, self.target_frame)[0]
                - self.center
            )
            self.new_out /= np.linalg.norm(self.new_out)
        frame, center = self.old_frame, self.old_center
        opening = smooth((t - 8.5) / 0.7)
        phase = "Release operating hand"
        if t >= 9.2:
            center = self.old_center + 0.1 * self.old_out * smooth((t - 9.2) / 0.8)
            phase = "Retract operating hand"
        if t >= 10:
            fraction = smooth((t - 10) / 1.5)
            frame = Slerp(
                [0, 1], Rotation.from_matrix([self.old_frame, self.target_frame])
            )([fraction]).as_matrix()[0]
            direction = (1 - fraction) * self.old_out + fraction * self.new_out
            direction /= np.linalg.norm(direction)
            center = (
                (1 - fraction) * self.old_center
                + fraction * self.target_center
                + 0.1 * direction
            )
            phase = "Move to front face"
        if t >= 11.5:
            frame = self.target_frame
            center = self.target_center + 0.1 * self.new_out * (
                1 - smooth((t - 11.5) / 1)
            )
            phase = "Approach front face"
        if t >= 12.5:
            opening = 1 - smooth((t - 12.5) / 1)
            phase = "Close front-face grasp"
        if t >= 13.5:
            phase = "Settle front-face grasp"
        if t >= 14.5:
            angle = -np.pi / 2 * smooth((t - 14.5) / 4)
            frame = (
                self.face_frame @ Rotation.from_euler("z", angle).as_matrix() @ self.sr
            )
            center = self.center + 0.01905 * (self.face_frame[:, 2] - frame[:, 2])
            phase = "Turn front face" if t < 18.5 else "Verify two-move sequence"
        joints = {
            name: self.closed[name] + opening * (self.open[name] - self.closed[name])
            for name in self.closed
        }
        return frame, center, joints, phase.replace("front", self.face)
