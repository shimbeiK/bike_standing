"""アクションモジュール (Actions).

ポリシーが出力するアクションの設定を定義するクラス群。
アクション設定は ``cfg.actions`` に辞書として登録する。

アクション一覧:
    BikeActions.build() が返す辞書:
        "steering" : ステアリング角度制御（バランス制御中は固定・scale=0）
        "drive"    : 後輪トルク制御（バランスの主制御）
"""

import math
from typing import Dict

### 変更点ここから: PID制御用に必要なモジュールを追加インポート ###
import torch
from dataclasses import dataclass
from mjlab.managers.action_manager import ActionTerm
### 変更点ここまで ###

from mjlab.envs.mdp.actions import JointEffortActionCfg, JointPositionActionCfg

### 変更点ここから: 速度型PID制御用のアクションクラスと設定データクラスを追加 ###
class WheelVelocityPidAction(ActionTerm):
    """後輪の目標速度に対して、速度型PID制御でトルクを計算・適用するカスタムアクション."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # ご指定のPIDゲイン
        self.kp = 0.48
        self.ki = 0.0086
        self.kd = 0.0

        # 対象アセットとジョイントの取得
        self.asset = self.env.scene[self.cfg.entity_name]
        self.joint_ids, self.joint_names = self.asset.find_joints(self.cfg.actuator_names)
        self.num_joints = len(self.joint_ids)

        # PID内部状態の初期化 [B, num_joints]
        self.prev_error = torch.zeros(self.env.num_envs, self.num_joints, device=self.env.device)
        self.prev_prev_error = torch.zeros(self.env.num_envs, self.num_joints, device=self.env.device)
        self.prev_output = torch.zeros(self.env.num_envs, self.num_joints, device=self.env.device)

    def reset(self, env_ids=None):
        """環境リセット時にPIDの内部状態をクリアする."""
        if env_ids is None:
            self.prev_error.fill_(0.0)
            self.prev_prev_error.fill_(0.0)
            self.prev_output.fill_(0.0)
        else:
            self.prev_error[env_ids] = 0.0
            self.prev_prev_error[env_ids] = 0.0
            self.prev_output[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        """ポリシーからの出力を目標速度として保持する."""
        self.target_velocities = actions * self.cfg.scale

    def apply_actions(self):
        """目標速度と現在速度の偏差からPIDでトルクを計算し、適用する."""
        # 現在のジョイント速度を取得
        current_vel = self.asset.data.joint_vel[:, self.joint_ids]

        # 偏差の計算 (目標速度 - 現在速度)
        error = self.target_velocities - current_vel

        # 速度型PIDの操作量増分（Delta u）の計算
        delta_u = (
            self.kp * (error - self.prev_error) +
            self.ki * error +
            self.kd * (error - 2 * self.prev_error + self.prev_prev_error)
        )

        # 操作量の更新 (u_t = u_{t-1} + Delta u)
        output = self.prev_output + delta_u

        # 内部状態の更新
        self.prev_prev_error = self.prev_error.clone()
        self.prev_error = error.clone()
        self.prev_output = output.clone()

        # アセットへのトルク（effort）適用
        self.asset.set_joint_effort_target(output, joint_ids=self.joint_ids)


@dataclass
class WheelVelocityPidActionCfg:
    """WheelVelocityPidAction用のアクション設定データクラス."""
    class_type: type = WheelVelocityPidAction
    entity_name: str = "robot"
    actuator_names: tuple = ()
    scale: float = 1.0
### 変更点ここまで ###


class BikeActions:
    """バイクバランス環境のアクション設定をまとめたクラス.

    インスタンス化せずに build() クラスメソッドを呼び出して使う。
    """

    # ステアリング設定
    STEERING_JOINT = "fork_yaw"
    STEERING_SCALE = 0.0         # バランス制御中はステアリング固定
    STEERING_USE_OFFSET = True   # 初期角度 (60 deg) をオフセットとして保持

    # ドライブ設定
    DRIVE_JOINT = "back_tire_pitch"
    ### 変更点ここから: スケールの意味が「トルク」から「目標速度 [rad/s]」に変わるため値を調整 ###
    # エージェントの出力(-1.0 ~ 1.0)が、最大何rad/sの目標速度になるかを指定します。
    # 元の4.0だと最大4rad/s(ゆっくり)になってしまうため、仮に20.0に変更しています。
    DRIVE_SCALE = 20.0           # 目標速度スケール [rad/s / normalized_action]
    ### 変更点ここまで ###

    @classmethod
    def build(cls) -> Dict[str, object]:
        """アクション設定辞書を生成して返す.

        Returns:
            cfg.actions に渡す辞書::

                {
                    "steering": JointPositionActionCfg(...),
                    "drive"   : JointEffortActionCfg(...), # ← 元のコメント維持
                }

        Notes:
            steering の scale=0 はステアリングを固定することを意味する。
            バランス制御はすべて後輪トルク (drive) で行う。
        """
        return {
            "steering": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(cls.STEERING_JOINT,),
                scale=cls.STEERING_SCALE,
                use_default_offset=cls.STEERING_USE_OFFSET,
            ),
            ### 変更点ここから: JointEffortActionCfg を WheelVelocityPidActionCfg に差し替え ###
            "drive": WheelVelocityPidActionCfg(
                entity_name="robot",
                actuator_names=(cls.DRIVE_JOINT,),
                scale=cls.DRIVE_SCALE,
            ),
            ### 変更点ここまで ###
        }