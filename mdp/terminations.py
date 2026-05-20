from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def roll_angle_limit(
  env: ManagerBasedRlEnv,
  threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """
  機体のロール角（横傾き）が指定した閾値（ラジアン）を超えた場合に終了(True)を返す。
  """
  asset: Entity = env.scene[asset_cfg.name]
  
  # ルート（車体）のクォータニオンと重力ベクトルを取得 [B, 4], [B, 3]
  body_quat_w = asset.data.root_link_quat_w
  gravity_w = asset.data.gravity_vec_w

  # 重力ベクトルをローカル（車体）座標系に投影 [B, 3]
  # 平地の場合、重力は[0, 0, -1]。車体がロール角phiだけ傾くと、ローカルのY成分は sin(phi) となる。
  projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
  
  # Y成分からロール角の絶対値を逆算
  roll_sin = torch.abs(projected_gravity_b[:, 1])
  roll_angle = torch.asin(torch.clamp(roll_sin, min=-1.0, max=1.0))
  
  # 閾値を超えているか判定 [B]
  return roll_angle > threshold


def position_limit(
  env: ManagerBasedRlEnv,
  limit_x: float,
  limit_y: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """
  機体が初期位置から指定したXY座標の範囲（メートル）を越えた場合に終了(True)を返す。
  """
  asset: Entity = env.scene[asset_cfg.name]
  
  # 車体のワールド座標XYを取得 [B, 2]
  root_xy_w = asset.data.root_link_pos_w[:, :2]
  
  # X軸またはY軸が閾値を超えているか判定 [B]
  out_of_x = torch.abs(root_xy_w[:, 0]) > limit_x
  out_of_y = torch.abs(root_xy_w[:, 1]) > limit_y
  
  return out_of_x | out_of_y


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  """
  最大ステップ数に到達した場合に終了(Truncate)を返す。
  ※ mjlabの環境クラスではエピソード長が自動管理されています。
  """
  # 現在のステップ数が、環境に設定された最大ステップ数以上になったか判定 [B]
  return env.episode_length_buf >= env.max_episode_length