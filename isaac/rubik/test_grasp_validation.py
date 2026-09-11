import unittest
from grasp_validation import evaluate_hold


class HoldValidationChecks(unittest.TestCase):
    def samples(self):
        return [
            dict(
                t=i * 0.02,
                core_position_error_m=0.002,
                core_rotation_error_deg=1.0,
                anchors_connected=True,
                joint_torque_max_nm=0.3,
                contact_loads_n=dict(left=[2.0], right=[2.0]),
            )
            for i in range(500)
        ]

    def test_supported_only_recording_cannot_pass(self):
        self.assertFalse(evaluate_hold(self.samples()[:100], 2, 10)["passed"])

    def test_stable_free_hold_passes(self):
        self.assertTrue(evaluate_hold(self.samples(), 2, 10)["passed"])

    def test_last_frame_recovery_cannot_hide_a_drop(self):
        s = self.samples()
        s[250]["core_position_error_m"] = 0.15
        self.assertFalse(evaluate_hold(s, 2, 10)["passed"])

    def test_one_hand_losing_contact_is_rejected(self):
        s = self.samples()
        for row in s[200:]:
            row["contact_loads_n"]["left"] = [0.0]
        self.assertFalse(evaluate_hold(s, 2, 10)["passed"])

    def test_jammed_finger_is_rejected(self):
        s = self.samples()
        s[250]["joint_torque_max_nm"] = 1.5
        self.assertFalse(evaluate_hold(s, 2, 10)["passed"])

    def test_missing_measurements_cannot_prove_a_continuous_hold(self):
        s = self.samples()
        self.assertFalse(evaluate_hold([s[100], s[-1]], 2, 10)["passed"])

    def test_nan_pose_is_rejected(self):
        s = self.samples()
        s[250]["core_position_error_m"] = float("nan")
        self.assertFalse(evaluate_hold(s, 2, 10)["passed"])


if __name__ == "__main__":
    unittest.main()
