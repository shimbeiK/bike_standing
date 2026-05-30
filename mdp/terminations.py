"""終了条件モジュール (Terminations).

エピソードの終了判定関数を定義するクラス群。
各クラスは static メソッドとして終了関数を持ち、
TerminationTermCfg(func=BikeTerminations.fell_over, ...) のように登録して使う。

終了条件一覧:
    BikeTerminations.fell_over      : 転倒角度が閾値を超えた場合
    BikeTerminations.out_of_bounds  : 後輪の累積回転量が範囲外に出た場合
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
        limit: float=0.5,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        """後輪のオドメトリ（累積回転量）が limit を超えたら True を返す."""
        from mjlab.tasks.bike_standing.mdp.observations import BikeObservations
        
        # 1. 全環境分のオドメトリを取得 (Shape: [N, 1])
        odom = BikeObservations.wheel_odometry(env)
        
        # 2. squeeze(-1) で余分な次元を削る (Shape: [N])
        odom_1d = odom.squeeze(-1)
        
        # 3. 絶対値をとって前後両方の範囲外を判定 (戻り値 Shape: [N])
        return torch.abs(odom_1d) > limit

    @staticmethod
    def print_observations(env, interval: int = 50) -> torch.Tensor:
        """推論（Play）時にターミナルへ観測値を出力するためのダミー終了条件.
        
        指定インターバルごとに観測値をPrintするが、終了判定としては常にFalseを返す。
        """
        # 環境インスタンスに直接ステップカウンタを付与して保持する
        if not hasattr(env, "_play_print_count"):
            env._play_print_count = 0

        if env._play_print_count % interval == 0:
            # 循環インポートを防ぐため関数内でインポート
            from mjlab.tasks.bike_standing.mdp.observations import BikeObservations
            
            # 各観測関数を呼び出して、バッチの先頭（0番目の環境）の値を取得
            # 戻り値は [N, 1] のテンソルなので、[0] で取り出して .item() で数値化
            roll = BikeObservations.base_roll(env)[0].item()
            gyro = BikeObservations.base_gyro(env)[0].item()
            odom = BikeObservations.wheel_odometry(env)[0].item()

            print(f"Step {env._play_print_count:04d} | Roll: {roll:+5.2f} rad | Gyro: {gyro:+5.2f} rad/s | Odom: {odom:+5.2f} rad")

        env._play_print_count += 1

        # 「誰も終了条件を満たしていない」という意味で、すべてFalseのテンソルを返す
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)