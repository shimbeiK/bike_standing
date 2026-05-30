"""観測モジュール (Observations).

ポリシーへ渡す観測量を定義するクラス群。
各クラスは static メソッドとして観測関数を持ち、
ObservationTermCfg(func=BikeObservations.base_roll) のように登録して使う。

観測一覧:
    BikeObservations.base_roll       : 本体の転倒角度  [rad]         Shape: [N, 1]
    BikeObservations.base_gyro       : 本体の転倒角速度 [rad/s]       Shape: [N, 1]
    BikeObservations.wheel_odometry  : 後輪の累積回転量 [rad]         Shape: [N, 1]
    BikeObservations.wheel_velocity  : 後輪の回転角速度 [rad/s]       Shape: [N, 1]
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
    ※注意: 状態を持つ観測（ジャイロ、オドメトリ）を並列環境で正しく機能させるため、
    環境のリセットイベント時に必ず `BikeObservations.reset_states` を呼び出してください。
    """

    @staticmethod
    def reset_states(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
        """非同期リセット対応: エピソードが終了した環境の内部状態のみをゼロクリアする.

        Args:
            env    : 環境インスタンス
            env_ids: リセット対象となる環境のインデックス (Shape: [M])
        """
        # ジャイロのフィルタ状態のリセット
        if hasattr(env, "_gyro_filtered"):
            env._gyro_filtered[env_ids] = 0.0
            
        # オドメトリの累積状態のリセット
        if hasattr(env, "_wheel_odometry"):
            env._wheel_odometry[env_ids] = 0.0

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

        return roll.unsqueeze(1)

    @staticmethod
    def base_gyro(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        noise_param: float = 0.0,  # ガウスノイズの標準偏差 [deg/s]
        alpha: float = 0.98,       # ローパスフィルタの係数 (0-1)
    ) -> torch.Tensor:
        """本体の転倒角速度（ロール方向の角速度）を返す [rad/s].

        実機ジャイロに合わせてガウスノイズ (±0.4 deg/s) と
        一次ローパスフィルタ (α=0.7) を適用する。

        Args:
            env      : 環境インスタンス
            asset_cfg: 対象エンティティ設定

        Returns:
            フィルタ済み角速度 (Shape: [N, 1])
        """
        asset: Entity = env.scene[asset_cfg.name]
        raw_gyro = asset.data.root_link_ang_vel_b[:, 0]  # ボディ x 軸 = ロール方向

        noise = torch.randn_like(raw_gyro) * math.radians(noise_param)
        noisy_gyro = raw_gyro + noise

        # 内部状態が存在しない場合は、環境数分のゼロテンソルとして初期化
        if not hasattr(env, "_gyro_filtered"):
            env._gyro_filtered = torch.zeros(env.num_envs, device=env.device)
            
        # フィルタ更新 (Shape: [N])
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
        """
        asset: Entity = env.scene[asset_cfg.name]
        
        # 後輪ジョイントのインデックスを取得
        joint_ids = asset.find_joints(["back_tire_pitch"])[0]
        
        # 現在の後輪の角速度を取得 [N]
        wheel_vel = asset.data.joint_vel[:, joint_ids].squeeze(-1)

        # 1ステップあたりの時間 (dt) を計算
        dt = env.cfg.sim.mujoco.timestep * env.cfg.decimation
        
        # 内部状態が存在しない場合は、環境数分のゼロテンソルとして初期化
        if not hasattr(env, "_wheel_odometry"):
            env._wheel_odometry = torch.zeros(env.num_envs, device=env.device)
            
        # 角速度を積分して累積回転量を更新 (Shape: [N])
        env._wheel_odometry += wheel_vel * dt

        # 観測として返す (Shape: [N, 1])
        return env._wheel_odometry.unsqueeze(1)

    @staticmethod
    def wheel_velocity(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        """後輪の回転角速度を返す [rad/s].

        実機の速度センサのブレを考慮し、ガウスノイズを付加する。

        Args:
            env      : 環境インスタンス
            asset_cfg: 対象エンティティ設定

        Returns:
            後輪の角速度 (Shape: [N, 1])
        """
        asset: Entity = env.scene[asset_cfg.name]
        joint_ids = asset.find_joints(["back_tire_pitch"])[0]
        
        # 角速度の取得 [N]
        wheel_vel = asset.data.joint_vel[:, joint_ids].squeeze(-1)
        
        return wheel_vel.unsqueeze(1)