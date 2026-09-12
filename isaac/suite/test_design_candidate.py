import json
from pathlib import Path
import tempfile
import unittest
from design_candidate import load_candidate


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "candidate.json"
        self.data = {
            "schema": "atlas-course-design-v1",
            "design": {
                "name": "custom_course",
                "axis": "custom",
                "kv_factor": 0.8,
                "gear_factor": 1.2,
                "peak_current_factor": 1,
                "added_mass_factor": 0.4,
            },
        }

    def load(self):
        self.path.write_text(json.dumps(self.data))
        return load_candidate(self.path)

    def test_export_roundtrip(self):
        self.assertEqual(self.load(), self.data["design"])

    def test_invalid_inputs(self):
        for value in [float("nan"), float("inf"), True, -1, 2, "1"]:
            with self.subTest(value=value):
                self.data["design"]["kv_factor"] = value
                with self.assertRaises(ValueError):
                    self.load()

    def test_unimplemented_physics_rejected(self):
        self.data["design"]["added_rotor_inertia_factor"] = 1
        with self.assertRaises(ValueError):
            self.load()

    def test_bad_schema_rejected(self):
        self.data["schema"] = "another-robot"
        with self.assertRaises(ValueError):
            self.load()
