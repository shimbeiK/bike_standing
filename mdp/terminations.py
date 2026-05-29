"""終了条件モジュール (Terminations).

エピソードの終了判定関数を定義するクラス群。
各クラスは static メソッドとして終了関数を持ち、
TerminationTermCfg(func=BikeTerminations.fell_over, ...) のように登録して使う。

終了条件一覧:
    BikeTerminations.fell_over      : 転倒角度が閾値を超えた場合
    BikeTerminations.out_of_bounds  : XY 座標が範囲外に出た場合
"""

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .utils import compute_roll

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class BikeTerminations:
    """バイクバランス環境の終了条件関数をまとめたクラス.

    インスタンス化せずに static メソッドとして利用する。
    すべての関数は bool テンソル (Shape: [N]) を返す。
    """

    @staticmethod
    def fell_over(
        env: ManagerBasedRlEnv,
        limit_angle: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        """ロール角が limit_angle [rad] を超えたら True を返す.

        Args:
            env         : 環境インスタンス
            limit_angle : 転倒判定の閾値 [rad]
            asset_cfg   : 対象エンティティ設定

        Returns:
            転倒フラグ (Shape: [N], dtype=bool)
        """
        asset: Entity = env.scene[asset_cfg.name]
        roll = compute_roll(asset)
        return torch.abs(roll) > limit_angle

    @staticmethod
    def out_of_bounds(
        env: ManagerBasedRlEnv,
        limit: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        """XY 平面内でロボットが limit [m] を超えたら True を返す.

        ロボットの位置はエピソード原点からの相対座標で評価する。

        Args:
            env      : 環境インスタンス
            limit    : 許容移動距離 [m]
            asset_cfg: 対象エンティティ設定

        Returns:
            範囲外フラグ (Shape: [N], dtype=bool)
        """
        asset: Entity = env.scene[asset_cfg.name]
        root_pos_local = asset.data.root_link_pos_w - env.scene.env_origins
        root_xy = root_pos_local[:, :2]
        return (torch.abs(root_xy[:, 0]) > limit) | (torch.abs(root_xy[:, 1]) > limit)