"""Fetch MIT-licensed native Wuji USD assets at a pinned upstream revision."""

import json, subprocess, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

REV = "4c1073d0a3ad1daaf6546d219db751d8448d3888"
ROOT = Path(__file__).parent / "assets/wuji"
BASE = f"https://raw.githubusercontent.com/wuji-technology/wuji-description/{REV}/"
manifest = json.loads((ROOT / "provenance.json").read_text())
assert manifest["revision"] == REV
paths = list(manifest["sha256"])


def fetch(path):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-fsSL", "--max-time", "60", BASE + path, "-o", str(p)], check=True
    )
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != manifest["sha256"][path]:
        raise ValueError(f"Checksum mismatch: {path}")
    return path, digest


with ThreadPoolExecutor(max_workers=6) as pool:
    checks = dict(pool.map(fetch, paths))
(ROOT / "provenance.json").write_text(
    json.dumps(
        {
            "repository": "https://github.com/wuji-technology/wuji-description",
            "revision": REV,
            "sha256": checks,
        },
        indent=2,
    )
)
print("Downloaded", len(paths), "files")
