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

from mjlab.envs.mdp.actions import JointEffortActionCfg, JointPositionActionCfg


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
    DRIVE_SCALE = 4.0            # トルクスケール [N·m / normalized_action]

    @classmethod
    def build(cls) -> Dict[str, object]:
        """アクション設定辞書を生成して返す.

        Returns:
            cfg.actions に渡す辞書::

                {
                    "steering": JointPositionActionCfg(...),
                    "drive"   : JointEffortActionCfg(...),
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
            "drive": JointEffortActionCfg(
                entity_name="robot",
                actuator_names=(cls.DRIVE_JOINT,),
                scale=cls.DRIVE_SCALE,
            ),
        }