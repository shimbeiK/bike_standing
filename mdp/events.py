"""イベントモジュール (Events).

エピソードの開始・リセット時に実行されるイベント関数を定義するクラス群。
各クラスは static メソッドとしてイベント関数を持ち、
EventTermCfg(func=BikeEvents.reset_internal_state, mode="reset") のように登録して使う。

イベント一覧:
    BikeEvents.reset_internal_state : 内部状態（オドメトリ・ジャイロフィルタ）をリセット
"""

import torch

from mjlab.envs import ManagerBasedRlEnv


class BikeEvents:
    """バイクバランス環境のイベント関数をまとめたクラス.

    インスタンス化せずに static メソッドとして利用する。
    """

    @staticmethod
    def reset_internal_state(
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
    ) -> None:
        """エピソードリセット時に観測の内部積算状態をゼロクリアする.

        クリア対象:
            - ``env._wheel_odometry`` : ``BikeObservations.wheel_odometry`` が積分する
              後輪の累積回転量
            - ``env._gyro_filtered``  : ``BikeObservations.base_gyro`` が保持する
              ローパスフィルタ状態

        どちらも初回リセット前は存在しない場合があるため、
        ``hasattr`` で存在確認してから更新する。

        Args:
            env     : 環境インスタンス
            env_ids : リセット対象の環境インデックス (Shape: [M])
        """
        if hasattr(env, "_wheel_odometry"):
            env._wheel_odometry[env_ids] = 0.0

        if hasattr(env, "_gyro_filtered"):
            env._gyro_filtered[env_ids] = 0.0