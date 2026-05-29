"""bike_v3_standing パッケージ.

HBP V3 バイクのその場バランス制御環境を構成するモジュール群。

モジュール構成::

    bike_v3_standing/
    ├── __init__.py       : パッケージ公開インターフェース（このファイル）
    ├── utils.py          : クォータニオン演算・ロール角計算などの共通ユーティリティ
    ├── observations.py   : 観測関数  (BikeObservations)
    ├── rewards.py        : 報酬関数  (BikeRewards)
    ├── terminations.py   : 終了条件  (BikeTerminations)
    ├── events.py         : イベント  (BikeEvents)
    ├── actions.py        : アクション設定 (BikeActions)
    └── env_cfg.py        : 環境設定ファクトリ (bike_v3_env_cfg)

典型的な使い方::

    from bike_v3_standing import bike_v3_env_cfg

    cfg = bike_v3_env_cfg(play=False)
"""

from .actions import BikeActions
from .events import BikeEvents
from .observations import BikeObservations
from .rewards import BikeRewards
from .terminations import BikeTerminations
from .utils import compute_roll, quat_apply_inverse

__all__ = [
    "BikeObservations",
    "BikeRewards",
    "BikeTerminations",
    "BikeEvents",
    "BikeActions",
    "compute_roll",
    "quat_apply_inverse",
]