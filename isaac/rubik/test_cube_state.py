import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from cube_state import *
from wuji_kinematics import Hand


class CubeChecks(unittest.TestCase):
    def test_all_moves_have_inverse(self):
        for face in FACES:
            for suffix in ("", "'", "2"):
                move = face + suffix
                self.assertTrue(
                    decode(state_after(move + " " + inverse(move)))["solved"]
                )
                self.assertFalse(decode(state_after(move))["solved"])

    def test_scramble(self):
        self.assertFalse(decode(state_after(SCRAMBLE))["solved"])
        self.assertTrue(
            decode(state_after(SCRAMBLE + " " + inverse(SCRAMBLE)))["solved"]
        )

    def test_reject_incomplete_turn(self):
        state = state_after("")
        mask = COORDS[:, 2] == 1
        state[mask] = Rotation.from_rotvec([0, 0, -np.pi / 4]).as_matrix() @ state[mask]
        self.assertFalse(decode(state)["aligned"])

    def test_reject_misaligned_cubie(self):
        state = state_after("")
        state[0] = Rotation.from_rotvec([0.2, 0, 0]).as_matrix()
        self.assertFalse(decode(state)["aligned"])

    def test_grasp_reachable(self):
        for side in ("left", "right"):
            hand = Hand(side)
            for closure in (-0.005, 0.0003, 0.003):
                self.assertLess(hand.grasp(closure)[1], 0.0001)


if __name__ == "__main__":
    unittest.main()
