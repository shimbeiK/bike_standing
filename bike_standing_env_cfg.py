"""Base configuration for bicycle balancing tasks.

This module provides a factory function to create a base bicycle balancing task config.
Robot-specific configurations call the factory and customize as needed.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg

# mdp関数群のインポート (プロジェクトの構造に合わせてください)
from mjlab.tasks.bike_standing import mdp 


def make_bike_v3_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base bicycle balancing task configuration."""

  ##
  # Observations
  ##

  actor_terms = {
    # "base_roll": ObservationTermCfg(func=mdp.BikeObservations.base_roll),
    # "base_gyro": ObservationTermCfg(func=mdp.BikeObservations.base_gyro),
    # "wheel_odometry": ObservationTermCfg(
    #   func=mdp.BikeObservations.wheel_odometry,
    #   params={"joint_name": ""}, # [Override] サブクラスで指定
    # ),
  }

  observations = {
    # "actor": ObservationGroupCfg(
    #   terms=actor_terms,
    #   concatenate_terms=True,
    #   enable_corruption=True,
    # ),
    # "critic": ObservationGroupCfg(
    #   terms=actor_terms, # Criticにも同じ情報を与える
    #   concatenate_terms=True,
    #   enable_corruption=False,
    # ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    # "steering": JointPositionActionCfg(
    #   entity_name="robot",
    #   actuator_names=(), # [Override] サブクラスで指定
    #   scale=1.0,         # [Override] サブクラスで指定
    #   use_default_offset=False,
    # ),
    # "drive": JointVelocityActionCfg(
    #   entity_name="robot",
    #   actuator_names=(), # [Override] サブクラスで指定
    #   scale=4.0,         # [Override] サブクラスで指定
    #   use_default_offset=False,
    # )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {} # 静止バランスのため空

  ##
  # Events
  ##

#   events = {
#     "randomize_mass": EventTermCfg(func=envs_mdp.randomize_mass, mode="startup"),
#     "randomize_friction": EventTermCfg(func=mdp.randomize_friction, mode="startup"),
#     "randomize_damping": EventTermCfg(func=mdp.randomize_damping, mode="startup"),
#     "randomize_com": EventTermCfg(func=mdp.randomize_com, mode="startup"),
#     "randomize_armature": EventTermCfg(func=mdp.randomize_armature, mode="startup"),
#     "apply_sim_randomization": EventTermCfg(func=mdp.apply_randomization_to_sim, mode="startup"),
#     "reset_base": EventTermCfg(
#       func=mdp.reset_root_state_uniform,
#       mode="reset",
#       params={
#         "pose_range": {"roll": (-math.radians(2.0), math.radians(2.0))},
#       },
#     ),
#   }

  ##
  # Rewards
  ##

  rewards = {
    # "upright": RewardTermCfg(
    #   func=mdp.BikeRewards.upright,
    #   weight=2.0,
    #   params={"std": math.sqrt(0.25)},
    # ),
    # "odometry_penalty": RewardTermCfg(
    #   func=mdp.BikeRewards.odometry_penalty,
    #   weight=-8.0,
    #   params={"deadzone": 0.2},
    # )
  }

  ##
  # Terminations
  ##

  terminations = {
    # "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # "fell_over": TerminationTermCfg(
    #   func=mdp.BikeTerminations.roll_angle_limit,
    #   params={"threshold": math.radians(45.0)},
    # ),
    # "out_of_bounds": TerminationTermCfg(
    #   func=mdp.BikeTerminations.position_limit,
    #   params={"limit_x": 10.5, "limit_y": 10.5},
    # ),
  }

  ##
  # Curriculum
  ##

  curriculum = {}

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"), # 平地
      sensors=(), # デフォルトはセンサなし
      num_envs=4096, # 並列環境数
      env_spacing=3.0, # ★ここを追加！（各環境を3.0メートル間隔で格子状に配置する）
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    # events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot", # エンティティ名は基本的に "robot" で統一
      distance=3.0,
      elevation=-15.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.001),
    ),
    decimation=10,
    episode_length_s=20.0,
  )