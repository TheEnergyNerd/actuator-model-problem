"""atlas_actuators — physics-grounded actuator models for Isaac Lab.

Two fidelity tiers, one design dict:
  • envelope  (AtlasActuatorCfg)     — τ-ω/thermal clamp; cheapest; RL at scale
  • FOC       (AtlasActuatorFOCCfg)  — field-weakening voltage ellipse +
                                       current lag + thermal; the FOC-correct
                                       ceiling (the envelope is optimistic
                                       near base speed)

Both are parameterized straight from an Atlas design run's `controller` dict —
the same numbers that drive the emitted MCU firmware (sim == firmware, 6e-7).
"""
__version__ = "0.1.0"

from .foc_core import (
    FOCActuatorParams, foc_actuator_params, foc_refs,
    step_current_lag, step_dq,
)
# Simulator wrappers and the optional envelope tier are loaded only on access.
# Importing the FOC core or registering Allegro tasks must not require the
# separate motor-design checkout containing atlas_actuator.py.
_FOC_EXPORTS = {"AtlasActuatorFOC", "AtlasActuatorFOCCfg", "foc_cfg_kwargs"}
_ENVELOPE_EXPORTS = {"AtlasActuator", "AtlasActuatorCfg", "AtlasActuatorParams",
                     "compute_actuator_effort", "params_from_design", "actuator_cfg_kwargs"}


def __getattr__(name):
    from importlib import import_module
    if name in _FOC_EXPORTS:
        module = import_module(".foc_actuator", __name__)
    elif name in _ENVELOPE_EXPORTS:
        module = import_module(".envelope", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = ["FOCActuatorParams", "foc_actuator_params", "foc_refs",
           "step_current_lag", "step_dq", *sorted(_FOC_EXPORTS), *sorted(_ENVELOPE_EXPORTS)]
