"""Lazy registration keeps optional robot dependencies out of Allegro startup.

Isaac resolves config strings when a task is requested, exposing import errors.
"""
try:
    import gymnasium as gym
except ModuleNotFoundError as error:
    if error.name != "gymnasium":
        raise
    gym = None

_ENTRIES = [
    ('Isaac-Velocity-Flat-Anymal-C-Atlas-v0', "atlas_actuators.tasks.anymal_atlas:AnymalCFlatAtlasEnvCfg",
     "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg"),
    ('Isaac-Velocity-Flat-Anymal-C-AtlasFOC-v0', "atlas_actuators.tasks.anymal_atlas:AnymalCFlatAtlasFOCEnvCfg",
     "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg"),
    ('Isaac-Velocity-Flat-Anymal-C-AtlasFOC-G6-v0', "atlas_actuators.tasks.anymal_atlas:AnymalCFlatAtlasFOCG6EnvCfg",
     "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg"),
    ('Isaac-Velocity-Flat-Anymal-C-AtlasFOC-G12-v0', "atlas_actuators.tasks.anymal_atlas:AnymalCFlatAtlasFOCG12EnvCfg",
     "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg"),
    ('Isaac-Velocity-Flat-Anymal-C-HardPush-v0', "atlas_actuators.tasks.anymal_atlas:AnymalCFlatHardPushEnvCfg",
     "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg"),
    ('Isaac-Velocity-Flat-Anymal-C-AtlasFOC-Hard-v0', "atlas_actuators.tasks.anymal_atlas:AnymalCFlatAtlasFOCHardEnvCfg",
     "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg"),
    ('Isaac-Velocity-Flat-Anymal-C-AtlasFOC-G6-Sprint-v0', "atlas_actuators.tasks.anymal_atlas:AnymalCFlatAtlasFOCG6SprintEnvCfg",
     "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg"),
    ('Isaac-Velocity-Flat-Anymal-C-AtlasFOC-Sprint-v0', "atlas_actuators.tasks.anymal_atlas:AnymalCFlatAtlasFOCSprintEnvCfg",
     "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg"),
    ('Isaac-Velocity-Flat-Anymal-C-AtlasFOC-G12-Sprint-v0', "atlas_actuators.tasks.anymal_atlas:AnymalCFlatAtlasFOCG12SprintEnvCfg",
     "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg"),
    ('Isaac-Velocity-Flat-Anymal-C-AtlasFOC-Thermal-v0', "atlas_actuators.tasks.anymal_atlas:AnymalCFlatAtlasFOCThermalEnvCfg",
     "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg"),
    ('Isaac-Velocity-Flat-Anymal-C-AtlasFOC-ThermalObs-v0', "atlas_actuators.tasks.anymal_atlas:AnymalCFlatAtlasFOCThermalObsEnvCfg",
     "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg"),
    ('Isaac-Velocity-Flat-G1-AtlasFOC-v0', "atlas_actuators.tasks.g1_atlas:G1FlatAtlasFOCEnvCfg",
     "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg"),
    ('Isaac-Velocity-Flat-G1-AtlasFOC-Tuned-v0', "atlas_actuators.tasks.g1_atlas:G1FlatAtlasFOCTunedEnvCfg",
     "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg"),
    ('Isaac-Velocity-Flat-G1-Upright-v0', "atlas_actuators.tasks.g1_atlas:G1FlatUprightEnvCfg",
     "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg"),
    ('Isaac-Velocity-Flat-G1-AtlasFOC-Upright-v0', "atlas_actuators.tasks.g1_atlas:G1FlatAtlasFOCUprightEnvCfg",
     "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg"),
    ('Isaac-Velocity-Flat-G1-AtlasFOC-Upright-Hard-v0', "atlas_actuators.tasks.g1_atlas:G1FlatAtlasFOCUprightHardEnvCfg",
     "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg"),
    ('Isaac-Velocity-Flat-G1-AtlasFOC-Upright-HW-v0', "atlas_actuators.tasks.g1_atlas:G1FlatAtlasFOCUprightHWEnvCfg",
     "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg"),
    ('Isaac-Repose-Cube-Allegro-AtlasFOC-v0', "atlas_actuators.tasks.allegro_atlas:AllegroCubeAtlasFOCEnvCfg",
     "isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg"),
    ('Isaac-Repose-Cube-Allegro-AtlasFOC-Play-v0', "atlas_actuators.tasks.allegro_atlas:AllegroCubeAtlasFOCEnvCfg_PLAY",
     "isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg"),
    ('Isaac-Repose-Cube-Allegro-IdealPD-v0', "atlas_actuators.tasks.allegro_atlas:AllegroCubeIdealPDEnvCfg",
     "isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg"),
    ('Isaac-Repose-Cube-Allegro-AtlasFOC-F1-v0', "atlas_actuators.tasks.allegro_atlas:AllegroCubeF1EnvCfg",
     "isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg"),
    ('Isaac-Repose-Cube-Allegro-AtlasFOC-F2-v0', "atlas_actuators.tasks.allegro_atlas:AllegroCubeF2EnvCfg",
     "isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg"),
    ('Isaac-Repose-Cube-Allegro-AtlasFOC-F3-v0', "atlas_actuators.tasks.allegro_atlas:AllegroCubeF3EnvCfg",
     "isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg"),
    ('Isaac-Repose-Cube-Allegro-AtlasFOC-F4-v0', "atlas_actuators.tasks.allegro_atlas:AllegroCubeF4EnvCfg",
     "isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg"),
    ('Isaac-Repose-Cube-Allegro-AtlasFOC-F5-v0', "atlas_actuators.tasks.allegro_atlas:AllegroCubeF5EnvCfg",
     "isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg"),
    ('Isaac-Repose-Cube-Allegro-NaivePeak-v0', "atlas_actuators.tasks.allegro_atlas:AllegroCubeNaivePeakEnvCfg",
     "isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg"),
    ('Isaac-Repose-Cube-Allegro-GearedReal-v0', "atlas_actuators.tasks.allegro_atlas:AllegroCubeGearedRealEnvCfg",
     "isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg"),
    ('Isaac-Repose-Cube-Allegro-NaivePeak20-v0', "atlas_actuators.tasks.allegro_atlas:AllegroCubeNaivePeak20EnvCfg",
     "isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg"),
    ('Isaac-Repose-Cube-Allegro-GearedReal20-v0', "atlas_actuators.tasks.allegro_atlas:AllegroCubeGearedReal20EnvCfg",
     "isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg"),
    ('Isaac-Repose-Cube-Allegro-GearedReal20Obs-v0', "atlas_actuators.tasks.allegro_atlas:AllegroCubeGearedReal20ObsEnvCfg",
     "isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg"),
 ]

_RSL_ALLEGRO = "isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg"
_ENTRIES.extend([
    ("Isaac-Repose-Cube-Allegro-NaivePeak20-v1",
     "atlas_actuators.tasks.allegro_atlas:AllegroCubeNaivePeak20EnvCfg", _RSL_ALLEGRO),
    ("Isaac-Repose-Cube-Allegro-GearedReal20-v1",
     "atlas_actuators.tasks.allegro_atlas:AllegroCubeGearedReal20V1EnvCfg", _RSL_ALLEGRO),
    ("Isaac-Repose-Cube-Allegro-GearedReal20Obs-v1",
     "atlas_actuators.tasks.allegro_atlas:AllegroCubeGearedReal20ObsV1EnvCfg", _RSL_ALLEGRO),
])

_ENTRIES.append((
    "Isaac-Velocity-Flat-G1-AtlasFOC-Explicit-v1",
    "atlas_actuators.tasks.g1_atlas:G1FlatAtlasFOCExplicitEnvCfg",
    "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
))

_ENTRIES.append((
    "Isaac-Velocity-Flat-G1-AtlasFOC-Walking-v1",
    "atlas_actuators.tasks.g1_atlas:G1FlatAtlasWalkingEnvCfg",
    "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
))

_ENTRIES.append((
    "Isaac-Velocity-Flat-G1-AtlasFOC-Walking-v2",
    "atlas_actuators.tasks.g1_atlas:G1FlatAtlasGroundedWalkingEnvCfg",
    "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
))

_ENTRIES.append((
    "Isaac-Velocity-Flat-G1-AtlasFOC-Walking-v3",
    "atlas_actuators.tasks.g1_atlas:G1FlatAtlasDirectedWalkingEnvCfg",
    "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
))

_ENTRIES.append(("Isaac-Repose-Cube-Allegro-Precision-v1", "atlas_actuators.tasks.allegro_precision:AllegroPrecisionEnvCfg", _RSL_ALLEGRO))

if gym is not None:
    for task_id, config, agent in _ENTRIES:
        gym.register(
            id=task_id, entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": config, "rsl_rl_cfg_entry_point": agent},
        )
_TASKS_REGISTERED = gym is not None
_G1_REGISTERED = _TASKS_REGISTERED
