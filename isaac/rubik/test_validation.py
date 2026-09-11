import unittest
import numpy as np
from validation import measure_cube, stable_match


class ValidationChecks(unittest.TestCase):
    def test_detached_cube_is_not_accepted_as_solved(self):
        states = np.zeros((27, 13))
        states[:, 3] = 1
        self.assertTrue(measure_cube(states)["anchors_connected"])
        states[1, 0] = 0.002
        measured = measure_cube(states)
        self.assertTrue(measured["decoded"]["solved"])
        self.assertFalse(measured["anchors_connected"])

    def test_transient_match_is_not_a_stable_hold(self):
        samples = [
            dict(t=i * 0.02, anchors_connected=True, decoded=dict(facelets="expected"))
            for i in range(31)
        ]
        self.assertTrue(stable_match(samples, "expected"))
        samples[-2]["decoded"]["facelets"] = None
        self.assertFalse(stable_match(samples, "expected"))
        self.assertFalse(stable_match(samples[-1:], "expected"))

    def test_disconnected_anchor_breaks_hold(self):
        samples = [
            dict(t=i * 0.02, anchors_connected=True, decoded=dict(facelets="expected"))
            for i in range(31)
        ]
        samples[-2]["anchors_connected"] = False
        self.assertFalse(stable_match(samples, "expected"))


if __name__ == "__main__":
    unittest.main()
