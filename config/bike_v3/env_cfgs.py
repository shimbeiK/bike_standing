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
from pathlib import Path
import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.viewer import ViewerConfig

# ローカルエンティティ定義用の追加インポート
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlActuatorCfg

from mjlab.tasks.bike_standing.bike_standing_env_cfg import make_bike_v3_env_cfg
from mjlab.tasks.bike_standing.mdp.actions import BikeActions
from mjlab.tasks.bike_standing.mdp.events import BikeEvents
from mjlab.tasks.bike_standing.mdp.observations import BikeObservations
from mjlab.tasks.bike_standing.mdp.rewards import BikeRewards
from mjlab.tasks.bike_standing.mdp.terminations import BikeTerminations


# ======================================================================
# ローカルモデル (XML) の読み込み設定
# ※ディレクトリ階層に合わせて .parent の数を調整してください
# ======================================================================
_ASSETS_DIR = Path(__file__).parent.parent.parent / "config" / "bike_v3" / "assets"
_BIKE_XML   = _ASSETS_DIR / "bike_V3_mjcf.xml"

def _get_spec() -> mujoco.MjSpec:
    """MuJoCo モデルスペックをローカルXMLから読み込む。"""
    return mujoco.MjSpec.from_file(str(_BIKE_XML))

_BIKE_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        XmlActuatorCfg(target_names_expr=("back_tire_pitch",)),
        XmlActuatorCfg(target_names_expr=("fork_yaw",)),
    ),
)

_BIKE_INIT = EntityCfg.InitialStateCfg(
    joint_pos={
        "back_tire_pitch": 0.0,
        "fork_yaw": math.radians(60.0),
    },
    joint_vel={".*": 0.0},
)

def get_local_bike_v3_robot_cfg() -> EntityCfg:
    """ローカルのXMLファイルからロボットのEntityCfgを生成する"""
    return EntityCfg(
        spec_fn=_get_spec,
        articulation=_BIKE_ARTICULATION,
        init_state=_BIKE_INIT,
    )


# ======================================================================
# 環境設定ファクトリ本体
# ======================================================================
def bike_v3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """HBP V3 バイクのその場バランス環境設定を生成する."""
    
    # ------------------------------------------------------------------
    # 1. ベース設定の読み込み
    # ------------------------------------------------------------------
    cfg = make_bike_v3_env_cfg()

    # ------------------------------------------------------------------
    # 2. ロボットエンティティ (ローカル関数を呼び出すように変更)
    # ------------------------------------------------------------------
    robot_cfg = get_local_bike_v3_robot_cfg()
    robot_cfg.init_state.joint_pos = robot_cfg.init_state.joint_pos.copy()
    robot_cfg.init_state.joint_pos["^fork_yaw$"] = math.radians(60.0)
    cfg.scene.entities = {"robot": robot_cfg}

    # ------------------------------------------------------------------
    # 3. シミュレーション基本設定
    # ------------------------------------------------------------------
    cfg.sim.mujoco.timestep = 0.001
    cfg.decimation = 10
    cfg.episode_length_s = 50.0
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
    # 5. アクション
    # ------------------------------------------------------------------
    cfg.actions = BikeActions.build()

    # ------------------------------------------------------------------
    # 6. 観測
    # ------------------------------------------------------------------
    actor_terms = {
        "roll": ObservationTermCfg(
            func=BikeObservations.base_roll,
        ),
        "gyro": ObservationTermCfg(
            func=BikeObservations.base_gyro,
        ),
        "wheel_velocity": ObservationTermCfg(
            func=BikeObservations.wheel_velocity,
        ),
    }

    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    # ------------------------------------------------------------------
    # 7. 報酬
    # ------------------------------------------------------------------
    cfg.rewards.clear()
    cfg.rewards["upright"] = RewardTermCfg(
        func=BikeRewards.upright,
        weight=1.5,
    )
    cfg.rewards["odometry_penalty"] = RewardTermCfg(
        func=BikeRewards.odometry_penalty,
        weight=-.0,
        params={"max_odom": math.radians(4500.0)},
    )
    cfg.rewards["wheel_velocity_penalty"] = RewardTermCfg(
        func=BikeRewards.wheel_velocity_penalty,
        weight=-.01,
    )

    # ------------------------------------------------------------------
    # 8. 終了条件
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
    cfg.terminations["out_of_bounds"] = TerminationTermCfg(
        func=BikeTerminations.out_of_bounds,
        params={"limit": 0.5},
    )

    # ------------------------------------------------------------------
    # 9. イベント
    # ------------------------------------------------------------------
    cfg.events.clear()
    cfg.events = BikeEvents.build(
        # init_roll_range=(-math.radians(1.0), math.radians(1.0)),
        velocity_range=(-0.01, 0.01),
    ) 

    # ------------------------------------------------------------------
    # 10. コマンド / カリキュラム
    # ------------------------------------------------------------------
    cfg.commands = {}
    cfg.curriculum = {}

    # ------------------------------------------------------------------
    # 11. Play モード
    # ------------------------------------------------------------------
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.terminations["print_obs_debug"] = TerminationTermCfg(
            func=BikeTerminations.print_observations,
            params={"interval": 10},
        )
    return cfg