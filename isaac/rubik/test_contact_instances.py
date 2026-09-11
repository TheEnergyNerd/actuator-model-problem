import unittest
from pathlib import Path
from pxr import Usd, UsdPhysics
from contact_model import make_collisions_editable


class InstanceChecks(unittest.TestCase):
    def test_both_native_hands_have_editable_collision_shapes(self):
        for side in ("left", "right"):
            path = (
                Path(__file__).parent
                / f"assets/wuji/hand2/hand2_beta1/body/usd/{side}/wujihand2.usd"
            )
            stage = Usd.Stage.Open(str(path))
            self.assertEqual(
                sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse()), 0
            )
            shapes = make_collisions_editable(stage, f"/wujihand2_{side}")
            self.assertEqual(len(shapes), 21)
            self.assertTrue(all(not p.IsInstanceProxy() for p in shapes))
            self.assertEqual(
                len(make_collisions_editable(stage, f"/wujihand2_{side}")), 21
            )


if __name__ == "__main__":
    unittest.main()
