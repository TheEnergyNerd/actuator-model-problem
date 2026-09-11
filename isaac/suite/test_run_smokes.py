"""The batch runner must not promote a crash or timeout to a passing result."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RunnerTests(unittest.TestCase):
    def run_case(self, body, timeout=5):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher = root / "isaaclab.sh"
            launcher.write_text("#!/usr/bin/env python3\n" + body)
            launcher.chmod(0o755)
            output = root / "result"
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("run_smokes.py")),
                    "--isaaclab",
                    str(root),
                    "--output",
                    str(output),
                    "--tasks",
                    "peg",
                    "--timeout",
                    str(timeout),
                ],
                capture_output=True,
                text=True,
                timeout=25,
            )
            return (
                process.returncode,
                json.loads((output / "summary.json").read_text())[0],
            )

    def test_crash_without_result_fails(self):
        code, result = self.run_case("raise SystemExit(3)\n")
        self.assertEqual(code, 1)
        self.assertFalse(result["passed"])
        self.assertEqual(result["exit_code"], 3)

    def test_timeout_fails(self):
        code, result = self.run_case("import time\ntime.sleep(60)\n", timeout=0.1)
        self.assertEqual(code, 1)
        self.assertFalse(result["passed"])
        self.assertEqual(result["error"], "TimeoutExpired")

    def test_wrong_task_result_fails(self):
        code, result = self.run_case(
            "import sys, pathlib, json\n"
            "p = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
            "p.mkdir()\n"
            "(p / 'result.json').write_text(json.dumps({'passed': True, 'task': 'wrong'}))\n"
        )
        self.assertEqual(code, 1)
        self.assertFalse(result["passed"])

    def test_matching_success_passes(self):
        code, result = self.run_case(
            "import sys, pathlib, json\n"
            "p = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
            "p.mkdir()\n"
            "task = sys.argv[sys.argv.index('--task') + 1]\n"
            "(p / 'result.json').write_text(json.dumps({'passed': True, 'task': task}))\n"
        )
        self.assertEqual(code, 0)
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
