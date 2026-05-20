from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

def base_roll(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """機体のロール角を取得し、設定されていればノイズを付与する。"""
    asset: Entity = env.scene[asset_cfg.name]
    
    # ルートのクォータニオンと重力ベクトルからロール角を計算
    body_quat_w = asset.data.root_link_quat_w
    gravity_w = asset.data.gravity_vec_w
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
    
    roll_sin = torch.abs(projected_gravity_b[:, 1])
    roll_angle = torch.asin(torch.clamp(roll_sin, min=-1.0, max=1.0))

    # ノイズの付与
    if getattr(env.cfg.env, "real_syncro_noise", False):
        noise = torch.randn_like(roll_angle) * np.deg2rad(1.0)
        roll_angle += noise
        
    return roll_angle.unsqueeze(1) # [B, 1]

def base_gyro_filtered(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """IMUから角速度を取得し、ノイズ付与とローパスフィルタを適用する。"""
    asset: Entity = env.scene[asset_cfg.name]
    
    # ローカル座標系の角速度のX成分 (ロール軸の角速度) [B]
    raw_angular_vel = asset.data.root_link_ang_vel_b[:, 0]
    
    # ノイズの付与
    noisy_angular_vel = raw_angular_vel + torch.randn_like(raw_angular_vel) * np.deg2rad(0.4)
    
    # フィルタの更新 (env側に状態を保持している想定)
    alpha = getattr(env, "alpha", 0.7)
    if not hasattr(env, "filtered_gy"):
        env.filtered_gy = torch.zeros_like(noisy_angular_vel)
        
    env.filtered_gy = -(alpha * noisy_angular_vel + (1 - alpha) * env.filtered_gy)
    
    # クリップ処理
    if getattr(env.cfg.env, "real_syncro_noise", False):
        env.filtered_gy = torch.clamp(env.filtered_gy, min=-2.0, max=2.0)
        
    return env.filtered_gy.unsqueeze(1) # [B, 1]

def drive_velocity(
    env: ManagerBasedRlEnv,
    joint_name: str = "back_tire_pitch",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """後輪の回転速度を取得し、ノイズを付与する。"""
    asset: Entity = env.scene[asset_cfg.name]
    
    # 指定したジョイントのインデックスを取得
    joint_ids = asset.find_joints([joint_name])[0]
    
    # ジョイントの速度を取得 [B]
    wheel_vel = asset.data.joint_vel[:, joint_ids].squeeze(-1)
    
    # ノイズの付与
    noisy_drive_vel = wheel_vel + torch.randn_like(wheel_vel) * np.deg2rad(0.3)
    
    # 他の報酬計算等で使うために保存
    env.drive_vel = noisy_drive_vel
    
    return noisy_drive_vel.unsqueeze(1) # [B, 1]