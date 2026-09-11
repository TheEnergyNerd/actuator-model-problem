"""Cube state derived from cubie orientations, independent of any controller."""

import itertools
import numpy as np
from scipy.spatial.transform import Rotation

COORDS = np.array([c for c in itertools.product((-1, 0, 1), repeat=3) if any(c)])
FACES = "URFDLB"
NORMALS = np.array(
    [(0, 0, 1), (1, 0, 0), (0, -1, 0), (0, 0, -1), (-1, 0, 0), (0, 1, 0)]
)
RIGHT = np.array([(1, 0, 0), (0, 1, 0), (1, 0, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0)])
DOWN = np.array([(0, -1, 0), (0, 0, -1), (0, 0, -1), (0, 1, 0), (0, 0, -1), (0, 0, -1)])
SCRAMBLE = "R U F' L2 D B R' U2 F D'"


def inverse(sequence):
    return " ".join(
        m if m.endswith("2") else m[:-1] if m.endswith("'") else m + "'"
        for m in sequence.split()[::-1]
    )


def apply_move(orientations, move):
    if len(move) > 2 or move[0] not in FACES or move[1:] not in ("", "'", "2"):
        raise ValueError(f"Invalid move: {move}")
    normal = NORMALS[FACES.index(move[0])]
    angle = -np.pi / 2 * (2 if move.endswith("2") else -1 if move.endswith("'") else 1)
    result = orientations.copy()
    positions = np.einsum("nij,nj->ni", orientations, COORDS)
    mask = positions @ normal > 0.5
    result[mask] = Rotation.from_rotvec(normal * angle).as_matrix() @ result[mask]
    return result


def state_after(sequence):
    state = np.tile(np.eye(3), (26, 1, 1))
    for move in sequence.split():
        state = apply_move(state, move)
    return state


def decode(orientations, tolerance_deg=3):
    """Return facelets only at an aligned, nonduplicated state; never round a bad turn into success."""
    stickers = ["?"] * 54
    worst = 0.0
    for c, r in zip(COORDS, orientations, strict=True):
        p = r @ c
        if np.max(np.abs(p - np.round(p))) > np.sin(
            np.deg2rad(tolerance_deg)
        ) * np.sqrt(3):
            return {"aligned": False, "solved": False, "facelets": None}
        for f, n in enumerate(NORMALS):
            if c @ n != 1:
                continue
            v = r @ n
            face = int(np.argmax(NORMALS @ v))
            error = np.rad2deg(np.arccos(np.clip(NORMALS[face] @ v, -1, 1)))
            worst = max(worst, float(error))
            if error > tolerance_deg:
                return {"aligned": False, "solved": False, "facelets": None}
            row = int(round(p @ DOWN[face])) + 1
            col = int(round(p @ RIGHT[face])) + 1
            if not (0 <= row < 3 and 0 <= col < 3):
                return {"aligned": False, "solved": False, "facelets": None}
            idx = face * 9 + row * 3 + col
            if stickers[idx] != "?":
                return {"aligned": False, "solved": False, "facelets": None}
            stickers[idx] = FACES[f]
    facelets = "".join(stickers)
    aligned = "?" not in facelets
    return {
        "aligned": aligned,
        "solved": aligned and facelets == "".join(f * 9 for f in FACES),
        "facelets": facelets,
        "max_sticker_error_deg": worst,
    }
