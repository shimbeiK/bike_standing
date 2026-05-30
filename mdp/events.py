"""イベントモジュール (Events).

環境の初期化時やリセット時に発生するイベント（ランダマイズ等）を定義する。
"""

import math
from typing import Dict

import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

# フレームワーク標準のDomain Randomization関数はインポートして使用します
# ※パスがお使いの環境と異なる場合は微調整してください
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import (
    joint_vel_rel,
    reset_joints_by_offset,
    reset_scene_to_default,
    time_out,
    JointEffortActionCfg,
    # JointPositionActionCfg,  # fork を動かす場合はコメントアウトを解除
)
from mjlab.envs import ManagerBasedRlEnv

class BikeEvents:
    """バイクバランス環境のイベント設定とカスタムロジックをまとめたクラス."""

    @staticmethod
    def reset_internal_state(env, env_ids):
        # ======== 追加: ジャイロのフィルタ状態のリセット ========
        if hasattr(env, "_gyro_filtered"):
            env._gyro_filtered[env_ids] = 0.0
        # ========================================================
        # 2. オドメトリの基準位置を、リセットされた瞬間の現在位置で上書きする
        if hasattr(env, "_wheel_odometry"):
            # リセットされた環境(env_ids)の累積量をゼロに戻す
            env._wheel_odometry[env_ids] = 0.0
        
        # 転倒などで環境がリセットされた際、タイヤの基準角度をリセット時の角度で上書き
        if hasattr(env, "_initial_wheel_angle"):
            asset = env.scene["robot"]
            joint_ids = asset.find_joints(["back_tire_pitch"])[0]
            
            current_angle = asset.data.joint_pos[env_ids, joint_ids].squeeze(-1)
            env._initial_wheel_angle[env_ids] = current_angle.clone()
    
    @staticmethod
    def reset_wheel_odometry(env, env_ids):
        """リセットされた環境のオドメトリ内部状態をゼロに戻す."""
        # 最初のステップ実行前でまだ変数が作られていない場合のエラー回避
        if hasattr(env, "_reward_wheel_odometry"):
            # 終了した環境 (env_ids) だけを 0 にリセットする
            env._reward_wheel_odometry[env_ids] = 0.0
    

    @staticmethod
    def randomize_initial_tilt(env, env_ids, asset_cfg: SceneEntityCfg, roll_range: tuple[float, float]):
        """velocity_mdp に依存せず、ローカルで車体のRoll角（横傾き）をランダム化する.
        
        Args:
            env: 環境インスタンス
            env_ids: リセット対象の環境ID
            asset_cfg: 対象アセットの設定
            roll_range: ランダム化するRoll角の範囲 (min, max) [rad]
        """
        asset = env.scene[asset_cfg.name]
        
        # デフォルトのルート状態（位置3, 姿勢4, 線速度3, 角速度3）を取得 [B, 13]
        root_state = asset.data.default_root_state[env_ids].clone()
        
        # 指定範囲でランダムなRoll角を生成 [B]
        roll = torch.empty(len(env_ids), device=env.device).uniform_(*roll_range)
        
        # Roll角（X軸回転）をクォータニオン [w, x, y, z] に変換
        # qw = cos(θ/2), qx = sin(θ/2), qy = 0, qz = 0
        quat_w = torch.cos(roll / 2.0).unsqueeze(-1)
        quat_x = torch.sin(roll / 2.0).unsqueeze(-1)
        quat_y = torch.zeros_like(quat_x)
        quat_z = torch.zeros_like(quat_x)
        rand_quat = torch.cat([quat_w, quat_x, quat_y, quat_z], dim=-1)
        
        # 姿勢部分（インデックス3〜6）をランダム化したクォータニオンで上書き
        root_state[:, 3:7] = rand_quat
        
        # 物理シミュレーションに新しい状態を書き込む
        asset.write_root_state_to_sim(root_state, env_ids)

    @classmethod
    def build(
        cls,
        friction_range: tuple[float, float] = (1.0, 1.0),
        mass_range: tuple[float, float] = (1.0, 1.0),
        inertia_range: tuple[float, float] = (0, 0),
        init_roll_range: tuple[float, float] = (-math.radians(0.0), math.radians(0.0)),
        velocity_range: tuple[float, float] = (-0.0, 0.0),
    ) -> Dict[str, EventTermCfg]:
        """イベント設定辞書を生成して返す.
        
        引数でパラメータを渡すことで、env_cfgs.py側からランダマイズの幅などを上書きできます。
        """
        return {
            "randomize_friction": EventTermCfg(
                func=envs_mdp.dr.geom_friction,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "operation": "scale",
                    "ranges": friction_range,  # 引数を使用
                }
            ),
            "randomize_mass": EventTermCfg(
                func=envs_mdp.dr.body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "operation": "scale",
                    "ranges": mass_range,  # 引数を使用
                }
            ),
            "randomize_inertia": EventTermCfg(
                func=envs_mdp.dr.pseudo_inertia,
                mode="startup",
                params={
                    # body_names で適用対象を MJCFで定義された "body" という名前のパーツに限定
                    "asset_cfg": SceneEntityCfg("robot", body_names=["body"]),
                    
                    # 慣性モーメントは e^(2d) 倍になるため、math.log(scale) / 2.0 で逆算して渡す
                    "d_range": inertia_range,
                }
            ),
            "init_tilt": EventTermCfg(
                func=cls.randomize_initial_tilt, 
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "roll_range": init_roll_range,  # 引数を使用
                },
            ),
            "reset_internal_state": EventTermCfg(
                func=cls.reset_internal_state,
                mode="reset",
            ),
            "reset_back_tire": EventTermCfg(
            func=reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": velocity_range,
                "asset_cfg": SceneEntityCfg("robot", joint_names=("back_tire_pitch",)),
                },
            ),
            "reset_odom" : EventTermCfg(
            func=cls.reset_wheel_odometry,
            mode="reset",  # エピソードがリセットされるタイミングで実行
        ),
    }
