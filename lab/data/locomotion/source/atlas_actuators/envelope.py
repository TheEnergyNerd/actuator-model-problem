"""envelope.py — the envelope-tier actuator, re-exported from its canonical home.

The single source of truth is isaac/atlas_actuator.py (verified in-engine on an
L40S: ALL GREEN + the ANYmal demo). This shim loads it by file path so the
extension does not carry a drifting copy. Works for source checkouts and
editable installs (`pip install -e`, `./isaaclab.sh -i` style) — which is how
Isaac Lab extensions are used. A built wheel would need the file vendored;
that is intentionally not done silently.
"""
import importlib.util
import pathlib
import sys

_CANONICAL = pathlib.Path(__file__).resolve().parents[3] / "atlas_actuator.py"

if "atlas_actuator" in sys.modules:              # already imported elsewhere
    _mod = sys.modules["atlas_actuator"]
else:
    if not _CANONICAL.exists():
        raise ImportError(
            f"envelope tier: canonical isaac/atlas_actuator.py not found at "
            f"{_CANONICAL} — install the extension editable from the "
            f"motordesign repo checkout (pip install -e isaac/exts/atlas_actuators).")
    _spec = importlib.util.spec_from_file_location("atlas_actuator", _CANONICAL)
    _mod = importlib.util.module_from_spec(_spec)
    # register BEFORE exec: dataclasses resolve cls.__module__ via sys.modules
    sys.modules["atlas_actuator"] = _mod
    _spec.loader.exec_module(_mod)

AtlasActuatorParams = _mod.AtlasActuatorParams
compute_actuator_effort = _mod.compute_actuator_effort
params_from_design = _mod.params_from_design
actuator_cfg_kwargs = _mod.actuator_cfg_kwargs
AtlasActuator = _mod.AtlasActuator            # None when isaaclab absent
AtlasActuatorCfg = _mod.AtlasActuatorCfg      # None when isaaclab absent
