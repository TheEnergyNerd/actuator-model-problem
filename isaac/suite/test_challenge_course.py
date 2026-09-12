"""The policy height scanner must see exactly the physical course geometry."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from challenge_course import make_challenge, corridor_limit, stage_at


class ChallengeGeometryTest(unittest.TestCase):
    def test_sensor_surface_matches_every_collision_zone(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "course.usda"
            make_challenge(path)
            stage = Usd.Stage.Open(str(path))
            meshes = [UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
            sensor, *zones = meshes
            self.assertEqual(sensor.GetPrim().GetName(), "SensorSurface")
            self.assertEqual(sensor.ComputeVisibility(), "invisible")
            self.assertFalse(sensor.GetPrim().HasAPI(UsdPhysics.CollisionAPI))
            physical_vertices = []
            physical_faces = []
            offset = 0
            for mesh in zones:
                self.assertTrue(mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI))
                points = np.asarray(mesh.GetPointsAttr().Get())
                physical_vertices.extend(points)
                physical_faces.extend(
                    np.asarray(mesh.GetFaceVertexIndicesAttr().Get()) + offset
                )
                offset += len(points)
            np.testing.assert_allclose(sensor.GetPointsAttr().Get(), physical_vertices)
            np.testing.assert_array_equal(
                sensor.GetFaceVertexIndicesAttr().Get(), physical_faces
            )
            self.assertTrue(np.isfinite(physical_vertices).all())

    def test_stage_and_bridge_corridor(self):
        self.assertEqual(corridor_limit(9), 0.35)
        self.assertEqual(stage_at(17.5), "Recover from a sideways push")


if __name__ == "__main__":
    unittest.main()
