from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

def survival_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
  """倒れずに生き残っていることに対する継続ボーナス。"""
  # env.num_envs (並列環境数) 分の 1.0 のテンソルを返す
  return torch.ones(env.num_envs, device=env.device)


def upright_posture(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """機体のロール角（横傾き）をゼロに保つ報酬。"""
  asset: Entity = env.scene[asset_cfg.name]
  
  # ルート（車体）のクォータニオンと重力ベクトルを取得 [B, 4], [B, 3]
  body_quat_w = asset.data.root_link_quat_w
  gravity_w = asset.data.gravity_vec_w

  # 重力ベクトルをローカル（車体）座標系に投影 [B, 3]
  projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
  
  # Y軸方向の重力成分の2乗（これがロール傾きに相当します）
  roll_squared = torch.square(projected_gravity_b[:, 1])
  
  # ガウス関数(exp)で0に近いほど1（最大報酬）になるようスケーリング
  return torch.exp(-roll_squared / std**2)


def position_deviation_penalty(
  env: ManagerBasedRlEnv,
  deadzone: float = 0.2,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """その場に留まらず移動してしまった距離(オドメトリ)に対するペナルティ。"""
  asset: Entity = env.scene[asset_cfg.name]
  
  # 車体のワールド座標XY [B, 2]
  root_pos_xy = asset.data.root_link_pos_w[:, :2]
  
  # 原点からの距離 [B]
  distance = torch.norm(root_pos_xy, dim=-1)
  
  # デッドゾーン（許容範囲）を超えた分をコストとして返す
  cost = torch.clamp(distance - deadzone, min=0.0)
  return cost


def action_rate_penalty(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  """急激な操作（ステアリングやトルクの急変）に対するペナルティ。
  
  mjlabの標準的なアプローチとして、キューを手動で管理するのではなく、
  ActionManagerに保存されている前回のアクション(prev_action)を使用します。
  """
  # 現在の行動と前回の行動の差分 [B, num_actions]
  action_diff = env.action_manager.action - env.action_manager.prev_action
  
  # 差分の2乗和を返す [B]
  return torch.sum(torch.square(action_diff), dim=1)


def motor_effort_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """モータのトルク（消費エネルギー）に対するペナルティ。"""
  asset: Entity = env.scene[asset_cfg.name]
  
  # アクチュエータが実際に出力した力（トルク） [B, num_actuators]
  applied_effort = asset.data.applied_effort
  
  return torch.sum(torch.square(applied_effort), dim=1)