"""観測モジュール (Observations).

ポリシーへ渡す観測量を定義するクラス群。
各クラスは static メソッドとして観測関数を持ち、
ObservationTermCfg(func=BikeObservations.base_roll) のように登録して使う。

観測一覧:
    BikeObservations.base_roll       : 本体の転倒角度  [rad]         Shape: [N, 1]
    BikeObservations.base_gyro       : 本体の転倒角速度 [rad/s]       Shape: [N, 1]
    BikeObservations.wheel_odometry  : 後輪の累積回転量 [rad]         Shape: [N, 1]
"""

import math

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .utils import compute_roll

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class BikeObservations:
    """バイクバランス環境の観測関数をまとめたクラス.

    インスタンス化せずに static メソッドとして利用する。
    """

    @staticmethod
    def base_roll(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        """本体の転倒角度（ロール角）を返す [rad].

        実機傾斜計に合わせてガウスノイズ (±1 deg) を付加する。

        Args:
            env      : 環境インスタンス
            asset_cfg: 対象エンティティ設定

        Returns:
            ロール角 + ノイズ (Shape: [N, 1])
        """
        asset: Entity = env.scene[asset_cfg.name]
        roll = compute_roll(asset)

        noise = torch.randn_like(roll) * math.radians(1.0)
        return (roll + noise).unsqueeze(1)

    @staticmethod
    def base_gyro(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        """本体の転倒角速度（ロール方向の角速度）を返す [rad/s].

        実機ジャイロに合わせてガウスノイズ (±0.4 deg/s) と
        一次ローパスフィルタ (α=0.7) を適用する。
        フィルタ状態 ``env._gyro_filtered`` はイベントでリセットされる。

        Args:
            env      : 環境インスタンス
            asset_cfg: 対象エンティティ設定

        Returns:
            フィルタ済み角速度 (Shape: [N, 1])
        """
        asset: Entity = env.scene[asset_cfg.name]
        raw_gyro = asset.data.root_link_ang_vel_b[:, 0]  # ボディ x 軸 = ロール方向

        noise = torch.randn_like(raw_gyro) * math.radians(0.4)
        noisy_gyro = raw_gyro + noise

        alpha = 0.7
        if not hasattr(env, "_gyro_filtered"):
            env._gyro_filtered = torch.zeros_like(noisy_gyro)
        env._gyro_filtered = alpha * noisy_gyro + (1.0 - alpha) * env._gyro_filtered

        filtered = torch.clamp(env._gyro_filtered, min=-2.0, max=2.0)
        return filtered.unsqueeze(1)

    @staticmethod
    def wheel_odometry(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        """後輪の累積回転量（オドメトリ）を返す [rad].

        毎ステップの後輪角速度を積分して累積変位を求める。
        積分状態 ``env._wheel_odometry`` はイベントでリセットされる。

        Args:
            env      : 環境インスタンス
            asset_cfg: 対象エンティティ設定

        Returns:
            累積回転量 (Shape: [N, 1])
        """
        asset: Entity = env.scene[asset_cfg.name]
        joint_ids = asset.find_joints(["back_tire_pitch"])[0]
        wheel_vel = asset.data.joint_vel[:, joint_ids].squeeze(-1)  # [N]

        noise = torch.randn_like(wheel_vel) * math.radians(0.3)
        wheel_vel_noisy = wheel_vel + noise

        dt = env.cfg.sim.mujoco.timestep * env.cfg.decimation
        if not hasattr(env, "_wheel_odometry"):
            env._wheel_odometry = torch.zeros(env.num_envs, device=env.device)
        env._wheel_odometry += wheel_vel_noisy * dt

        return env._wheel_odometry.unsqueeze(1)