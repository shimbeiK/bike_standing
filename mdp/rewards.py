"""報酬モジュール (Rewards).

強化学習の報酬関数を定義するクラス群。
各クラスは static メソッドとして報酬関数を持ち、
RewardTermCfg(func=BikeRewards.upright, weight=4.0) のように登録して使う。

報酬一覧:
    BikeRewards.upright          : 転倒角度が小さいほど高報酬  [0, 1]
    BikeRewards.odometry_penalty : 後輪累積移動量のペナルティ  [0, ∞)
"""

import math

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .utils import compute_roll

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

def _tolerance(x: torch.Tensor, margin: float) -> torch.Tensor:
    return torch.exp(-0.5 * (x / margin) ** 2)

class BikeRewards:
    """バイクバランス環境の報酬関数をまとめたクラス.

    インスタンス化せずに static メソッドとして利用する。
    """

    @staticmethod
    def upright(
        env: ManagerBasedRlEnv,
        margin: float = math.radians(5.0), # marginに変更
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        roll = compute_roll(asset)
        # 線形からガウス型（_tolerance）に変更
        return _tolerance(roll, margin=margin)

    @staticmethod
    def odometry_penalty(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        max_odom: float = math.radians(1800.0),
    ) -> torch.Tensor:
        """後輪の累積回転量（オドメトリ）に対するペナルティ."""
        asset: Entity = env.scene[asset_cfg.name]
        joint_ids = asset.find_joints(["back_tire_pitch"])[0]

        # 現在の後輪の角速度を取得 [N]
        wheel_vel = asset.data.joint_vel[:, joint_ids].squeeze(-1)

        # 1ステップあたりの時間 (dt) を計算
        dt = env.cfg.sim.mujoco.timestep * env.cfg.decimation

        # 報酬用の内部状態が存在しない場合は初期化
        if not hasattr(env, "_reward_wheel_odometry"):
            env._reward_wheel_odometry = torch.zeros(env.num_envs, device=env.device)

        # オドメトリを更新
        env._reward_wheel_odometry += wheel_vel * dt

        # 上限でクリップして絶対値を返す
        odom_abs = torch.abs(env._reward_wheel_odometry)
        return torch.clamp(odom_abs, max=max_odom)

    @staticmethod
    def wheel_velocity_penalty(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        margin: float = math.radians(5.0),
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        joint_ids = asset.find_joints(["back_tire_pitch"])[0]

        # 現在の後輪の角速度を取得 [N]
        wheel_vel = asset.data.joint_vel[:, joint_ids].squeeze(-1)

        return 1.0 - _tolerance(wheel_vel, margin=margin)