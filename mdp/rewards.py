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


class BikeRewards:
    """バイクバランス環境の報酬関数をまとめたクラス.

    インスタンス化せずに static メソッドとして利用する。
    """

    @staticmethod
    def upright(
        env: ManagerBasedRlEnv,
        max_angle: float = math.radians(10.0),
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        """転倒角度が小さいほど高い報酬を返す.

        正立 (roll = 0) で 1.0、|roll| >= max_angle で 0.0 になるよう
        線形に正規化し、下限 0 でクランプする。

        報酬の計算式::

            r = clamp((max_angle - |roll|) / max_angle, min=0)

        Args:
            env      : 環境インスタンス
            max_angle: 転倒とみなすロール角の上限 [rad]（デフォルト: 10 deg）
            asset_cfg: 対象エンティティ設定

        Returns:
            正規化転倒報酬 (Shape: [N])
        """
        asset: Entity = env.scene[asset_cfg.name]
        roll = compute_roll(asset)
        normalized = (max_angle - torch.abs(roll)) / max_angle
        return torch.clamp(normalized, min=0.0)

    @staticmethod
    def odometry_penalty(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,  # noqa: ARG004
    ) -> torch.Tensor:
        """後輪の累積オドメトリ量をペナルティとして返す.

        その場に留まらせるため、後輪の累積回転量（絶対値）をコストとして返す。
        報酬重みは cfg 側で負符号（例: weight=-0.5）にする。

        ``env._wheel_odometry`` は ``BikeObservations.wheel_odometry`` が
        積分・更新する。リセットは ``BikeEvents.reset_internal_state`` が担う。

        Args:
            env      : 環境インスタンス
            asset_cfg: 未使用（シグネチャ統一のため保持）

        Returns:
            累積回転量の絶対値 (Shape: [N])
        """
        if not hasattr(env, "_wheel_odometry"):
            return torch.zeros(env.num_envs, device=env.device)
        return torch.abs(env._wheel_odometry)