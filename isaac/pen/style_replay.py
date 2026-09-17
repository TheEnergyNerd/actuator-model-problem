"""Display-only studio framing; never changes the recorded motion or collision geometry."""

import argparse, json
from pathlib import Path


def style(path):
    scene = json.loads(path.read_text())
    for m in scene["meshes"]:
        name = scene["bodies"][m["body"]]
        if "Hand/" in name:
            m["color"] = (
                [0.12, 0.18, 0.16] if "elastomer" in name else [0.72, 0.76, 0.69]
            )
        m["vertices"] = [round(v, 7) for v in m["vertices"]]
    scene["camera_offset"] = [0.0, -0.25, 0.17]
    scene["overview_camera"] = {"eye": [0.12, -0.32, 0.80], "target": [0.12, 0, 0.57]}
    scene["static_meshes"] = [
        dict(
            vertices=[-3, -3, 0.46, 3, -3, 0.46, 3, 3, 0.46, -3, 3, 0.46],
            indices=[0, 1, 2, 0, 2, 3],
            color=[0.83, 0.85, 0.81],
        )
    ]
    scene["display_note"] = (
        "Studio floor, display colors and camera only. Native mesh coordinates rounded to 0.1 micrometre; no pose edits."
    )
    path.write_text(json.dumps(scene, separators=(",", ":")))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scene", type=Path)
    a = p.parse_args()
    style(a.scene)
