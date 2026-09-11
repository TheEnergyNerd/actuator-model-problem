import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from passive import local_torques
from sequence_control import Sequence
from cube_state import SCRAMBLE, inverse, state_after, decode
from wuji_kinematics import Hand


class PassiveChecks(unittest.TestCase):
    def test_equal_opposite_torques(self):
        r = torch.tensor(Rotation.random(27, random_state=4).as_matrix())
        w = torch.tensor(np.random.default_rng(9).normal(size=(27, 3)))
        torque = local_torques(r, w)
        world = (r @ torque[..., None]).squeeze(-1)
        self.assertLess(float(world.sum(0).norm()), 1e-12)

    def test_force_is_negative_energy_gradient(self):
        r = Rotation.from_rotvec([0.24, -0.17, 0.31]).as_matrix()

        def energy(matrix):
            return -0.02 / 4 * np.sum(matrix**4) + 0.2 / 2 * (1 - np.max(matrix**2))

        grad = []
        eps = 1e-6
        for axis in np.eye(3):
            grad.append(
                (
                    energy(Rotation.from_rotvec(axis * eps).as_matrix() @ r)
                    - energy(Rotation.from_rotvec(-axis * eps).as_matrix() @ r)
                )
                / (2 * eps)
            )
        rots = torch.tensor(np.array([np.eye(3), r]))
        local = local_torques(rots, torch.zeros(2, 3), damping=0)
        np.testing.assert_allclose(r @ local[1].numpy(), -np.array(grad), atol=1e-10)

    def test_damping_removes_energy(self):
        r = torch.tensor(Rotation.random(27, random_state=4).as_matrix())
        w = torch.tensor(np.random.default_rng(9).normal(size=(27, 3)))
        tau = (r @ local_torques(r, w, detent=0, guide=0)[..., None]).squeeze(-1)
        self.assertLess(float((tau * w).sum()), 0)

    def test_sequence_split_and_continuity(self):
        sequence = Sequence(Hand("right"), inverse(SCRAMBLE))
        self.assertTrue(
            decode(state_after(SCRAMBLE + " " + " ".join(sequence.moves)))["solved"]
        )
        for boundary in (2, 3.5, 4.3, 8.3, 9.5, 10.3, 12, 13, 14):
            p0, r0, q0, _, _ = sequence.command(boundary - 1e-5)
            p1, r1, q1, _, _ = sequence.command(boundary + 1e-5)
            self.assertLess(np.linalg.norm(p1 - p0), 1e-5)
            self.assertLess(Rotation.from_matrix(r1 @ r0.T).magnitude(), 1e-4)
            self.assertLess(max(abs(q0[n] - q1[n]) for n in q0), 1e-4)

    def test_retry_reposition_starts_at_previous_retracted_pose(self):
        sequence = Sequence(Hand("right"), "U")
        previous = sequence.command(sequence.period - 1e-8)
        end_frame = (
            sequence.frames[0]
            @ Rotation.from_rotvec([0, 0, sequence.angles[0]]).as_matrix()
        )
        sequence.entry_frames[0] = end_frame
        sequence.frames[0] = (
            sequence.frames[0] @ Rotation.from_rotvec([0, 0, np.pi / 2]).as_matrix()
        )
        retry = sequence.command(0)
        np.testing.assert_allclose(previous[0], retry[0], atol=1e-8)
        np.testing.assert_allclose(previous[1], retry[1], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
