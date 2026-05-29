"""環境設定ファクトリ (env_cfg).

``bike_v3_env_cfg()`` が唯一のパブリック関数。
各サブモジュール（observations / rewards / terminations / events / actions）の
クラスを組み合わせて ManagerBasedRlEnvCfg を構築して返す。

使い方::

    from bike_v3_standing import bike_v3_env_cfg

    cfg = bike_v3_env_cfg(play=False)   # 学習用
    cfg = bike_v3_env_cfg(play=True)    # 推論・テスト用
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.viewer import ViewerConfig
from mjlab.asset_zoo.robots import get_bike_v3_robot_cfg
from mjlab.tasks.bike_standing.bike_standing_env_cfg import make_bike_v3_env_cfg

from mjlab.tasks.bike_standing.mdp.actions import BikeActions
from mjlab.tasks.bike_standing.mdp.events import BikeEvents
from mjlab.tasks.bike_standing.mdp.observations import BikeObservations
from mjlab.tasks.bike_standing.mdp.rewards import BikeRewards
from mjlab.tasks.bike_standing.mdp.terminations import BikeTerminations


def bike_v3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """HBP V3 バイクのその場バランス環境設定を生成する.

    観測 (ポリシー入力・次元数 = 3):
        roll            本体の転倒角度  [rad]         (Shape: [N, 1])
        gyro            本体の転倒角速度 [rad/s]       (Shape: [N, 1])
        wheel_odometry  後輪の累積回転量 [rad]         (Shape: [N, 1])

    報酬:
        upright          (+4.0)  転倒角度が小さいほど高報酬  [0, 1]
        odometry_penalty (−0.5)  後輪累積移動量のペナルティ  [0, ∞)

    終了条件:
        time_out        エピソード時間超過
        fell_over       転倒角度 ±10 deg 超過
        out_of_bounds   XY 座標 ±0.1 m 超過（play=True で無効化）

    Args:
        play: True のときは推論・テスト向けに設定を緩和する
              （エピソード無限・範囲外終了を無効化）

    Returns:
        組み立て済みの ManagerBasedRlEnvCfg
    """
    # ------------------------------------------------------------------
    # 1. ベース設定の読み込み
    # ------------------------------------------------------------------
    cfg = make_bike_v3_env_cfg()

    # ------------------------------------------------------------------
    # 2. ロボットエンティティ
    # ------------------------------------------------------------------
    robot_cfg = get_bike_v3_robot_cfg()
    robot_cfg.init_state.joint_pos = robot_cfg.init_state.joint_pos.copy()
    robot_cfg.init_state.joint_pos["^fork_yaw$"] = math.radians(60.0)
    cfg.scene.entities = {"robot": robot_cfg}

    # ------------------------------------------------------------------
    # 3. シミュレーション基本設定
    #    制御周期: timestep(1 ms) × decimation(10) = 10 ms
    # ------------------------------------------------------------------
    cfg.sim.mujoco.timestep = 0.001
    cfg.decimation = 10
    cfg.episode_length_s = 10.0
    cfg.sim.njmax = 200
    cfg.sim.nconmax = 100

    if cfg.scene.terrain is not None:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None

    # ------------------------------------------------------------------
    # 4. ビューア
    # ------------------------------------------------------------------
    cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_BODY
    cfg.viewer.entity_name = "robot"
    cfg.viewer.body_name = "body"
    cfg.viewer.distance = 3.0
    cfg.viewer.elevation = -15.0
    cfg.viewer.azimuth = 120.0

    # ------------------------------------------------------------------
    # 5. アクション  ← BikeActions.build() で一元管理
    # ------------------------------------------------------------------
    cfg.actions = BikeActions.build()

    # ------------------------------------------------------------------
    # 6. 観測  ← BikeObservations の static メソッドを登録
    # ------------------------------------------------------------------
    actor_terms = {
        "roll": ObservationTermCfg(
            func=BikeObservations.base_roll,
        ),
        "gyro": ObservationTermCfg(
            func=BikeObservations.base_gyro,
        ),
        "wheel_odometry": ObservationTermCfg(
            func=BikeObservations.wheel_odometry,
        ),
    }

    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,   # → [N, 3] に結合
            enable_corruption=False,
        ),
        "critic": ObservationGroupCfg(
            terms=actor_terms,        # Critic にも同じ観測を与える
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    # ------------------------------------------------------------------
    # 7. 報酬  ← BikeRewards の static メソッドを登録
    # ------------------------------------------------------------------
    cfg.rewards.clear()
    cfg.rewards["upright"] = RewardTermCfg(
        func=BikeRewards.upright,
        weight=4.0,
        params={"max_angle": math.radians(45.0)},
    )
    cfg.rewards["odometry_penalty"] = RewardTermCfg(
        func=BikeRewards.odometry_penalty,
        weight=-2.,
    )

    # ------------------------------------------------------------------
    # 8. 終了条件  ← BikeTerminations の static メソッドを登録
    # ------------------------------------------------------------------
    cfg.terminations.clear()
    cfg.terminations["time_out"] = TerminationTermCfg(
        func=lambda env: env.episode_length_buf >= env.max_episode_length,
        time_out=True,
    )
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=BikeTerminations.fell_over,
        params={"limit_angle": math.radians(10.0)},
    )
    # cfg.terminations["out_of_bounds"] = TerminationTermCfg(
    #     func=BikeTerminations.out_of_bounds,
    #     params={"limit": 0.1},
    # )

    # ------------------------------------------------------------------
    # 9. イベント  ← BikeEvents の static メソッドを登録
    # ------------------------------------------------------------------
    cfg.events.clear()

    # 初期傾き: ±1 deg のランダマイズ（mjlab 標準関数を利用）
    cfg.events["init_tilt"] = EventTermCfg(
        func=velocity_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "roll": (-math.radians(1.0), math.radians(1.0)),
            },
            "velocity_range": {},
        },
    )

    # オドメトリとジャイロフィルタの内部状態リセット
    cfg.events["reset_internal_state"] = EventTermCfg(
        func=BikeEvents.reset_internal_state,
        mode="reset",
    )

    # ------------------------------------------------------------------
    # 10. コマンド / カリキュラム（このタスクでは使用しない）
    # ------------------------------------------------------------------
    cfg.commands = {}
    cfg.curriculum = {}

    # ------------------------------------------------------------------
    # 11. Play モード（推論・テスト）向け上書き
    # ------------------------------------------------------------------
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.terminations.pop("out_of_bounds", None)

    return cfg