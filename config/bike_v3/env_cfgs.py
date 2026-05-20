"""HBP V3 Bicycle Environment in mjlab (IsaacLab) format."""

import math
import numpy as np
import torch
import mujoco

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.viewer import ViewerConfig
from mjlab.envs.mdp import dr
from mjlab.tasks.velocity import mdp as velocity_mdp
# 環境のベース設定関数と、ロボットのモデル設定関数をインポート
# （※パスはご自身の環境に合わせて適宜調整してください）
from mjlab.asset_zoo.robots import get_bike_v3_robot_cfg
from mjlab.tasks.bike_task.bike_standing_env_cfg import make_bike_v3_env_cfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# =====================================================================
# 1. MDP Functions (Observations, Rewards, Terminations, Events)
# =====================================================================
def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    クォータニオンの逆回転（ワールド座標 -> ローカル座標）をベクトルに適用します。
    
    Args:
        q (torch.Tensor): クォータニオン [w, x, y, z] 形式 (Shape: [N, 4])
        v (torch.Tensor): 変換する3次元ベクトル (Shape: [N, 3])
        
    Returns:
        torch.Tensor: 変換後の3次元ベクトル (Shape: [N, 3])
    """
    # スカラー部(w)とベクトル部(x, y, z)に分割
    w = q[..., 0:1]
    u = q[..., 1:4]
    
    # 逆回転のためのロドリゲスの回転公式ベースの計算
    t = 2.0 * torch.cross(u, v, dim=-1)
    v_prime = v - w * t + torch.cross(u, t, dim=-1)
    
    return v_prime

def obs_base_roll(env: ManagerBasedRlEnv, asset_cfg=_DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    gravity_b = quat_apply_inverse(asset.data.root_link_quat_w, asset.data.gravity_vec_w)
    roll_sin = torch.abs(gravity_b[:, 1])
    roll = torch.asin(torch.clamp(roll_sin, min=-1.0, max=1.0))
    # real_syncro_noise (±1deg)
    noise = torch.randn_like(roll) * math.radians(1.0)
    return (roll + noise).unsqueeze(1)

def obs_base_gyro(env: ManagerBasedRlEnv, asset_cfg=_DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    raw_gyro = asset.data.root_link_ang_vel_b[:, 0]
    # noise (±0.4deg/s相当)
    noise = torch.randn_like(raw_gyro) * math.radians(0.4)
    noisy_gyro = raw_gyro + noise
    
    # ローパスフィルタ
    alpha = 0.7
    if not hasattr(env, "filtered_gy"):
        env.filtered_gy = torch.zeros_like(noisy_gyro)
    env.filtered_gy = -(alpha * noisy_gyro + (1 - alpha) * env.filtered_gy)
    env.filtered_gy = torch.clamp(env.filtered_gy, min=-2.0, max=2.0)
    return env.filtered_gy.unsqueeze(1)

def obs_drive_vel(env: ManagerBasedRlEnv, asset_cfg=_DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    joint_ids = asset.find_joints(["back_tire_pitch"])[0]
    wheel_vel = asset.data.joint_vel[:, joint_ids].squeeze(-1)
    noise = torch.randn_like(wheel_vel) * math.radians(0.3)
    return (wheel_vel + noise).unsqueeze(1)

def rew_survival(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)

def rew_upright(env: ManagerBasedRlEnv, asset_cfg=_DEFAULT_ASSET_CFG) -> torch.Tensor:
    # 45度(pi/4)を基準とした正規化報酬
    asset: Entity = env.scene[asset_cfg.name]
    gravity_b = quat_apply_inverse(asset.data.root_link_quat_w, asset.data.gravity_vec_w)
    roll = torch.asin(torch.clamp(torch.abs(gravity_b[:, 1]), min=-1.0, max=1.0))
    max_angle = math.radians(45.0)
    normalized_tilt = (max_angle - roll) / max_angle
    return torch.clamp(normalized_tilt, min=0.0)

def rew_odometry(env: ManagerBasedRlEnv, deadzone: float = 0.2, asset_cfg=_DEFAULT_ASSET_CFG) -> torch.Tensor:
    # 初期位置からの移動距離に対するペナルティ
    asset: Entity = env.scene[asset_cfg.name]
    dist = torch.norm(asset.data.root_link_pos_w[:, :2], dim=-1)
    return torch.clamp(dist - deadzone, min=0.0)

def rew_torque(env: ManagerBasedRlEnv, asset_cfg=_DEFAULT_ASSET_CFG) -> torch.Tensor:
    # 前回の行動との差分(unstable)と、絶対値(abs)のペナルティ
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    
    normalized_diff = torch.abs(action[:, 1] - prev_action[:, 1])
    normalized_abs = torch.abs(action[:, 1])
    return normalized_diff + normalized_abs

def term_fell_over(env: ManagerBasedRlEnv, limit_angle: float, asset_cfg=_DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    gravity_b = quat_apply_inverse(asset.data.root_link_quat_w, asset.data.gravity_vec_w)
    roll = torch.asin(torch.clamp(torch.abs(gravity_b[:, 1]), min=-1.0, max=1.0))
    return roll > limit_angle

def term_out_of_bounds(env: ManagerBasedRlEnv, limit: float, asset_cfg=_DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    root_xy = asset.data.root_link_pos_w[:, :2]
    return (torch.abs(root_xy[:, 0]) > limit) | (torch.abs(root_xy[:, 1]) > limit)

# =====================================================================
# 2. Configuration Factory
# =====================================================================

def bike_v3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create HBP V3 Bicycle environment configuration."""
    
    # 1. ベースとなる設定を読み込む
    cfg = make_bike_v3_env_cfg()

    # 2. ロボットエンティティの登録
    cfg.scene.entities = {"robot": get_bike_v3_robot_cfg()}

    # 3. シミュレーションと地形のオーバーライド
    cfg.sim.mujoco.timestep = 0.01
    cfg.decimation = 10
    cfg.episode_length_s = 20.0
    cfg.sim.njmax = 200
    cfg.sim.nconmax = 100  
    
    if cfg.scene.terrain is not None:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None

    # 4. ビューアーのオーバーライド
    cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_BODY
    cfg.viewer.entity_name = "robot"
    cfg.viewer.distance = 3.0
    cfg.viewer.elevation = -15.0
    cfg.viewer.azimuth = 120.0

    # 5. Actionsのオーバーライド
    cfg.actions = {
        "steering": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("fork_yaw",),
            scale=math.radians(80.0),
            use_default_offset=False,
        ),
        "drive": JointVelocityActionCfg(
            entity_name="robot",
            actuator_names=("back_tire_pitch",),
            scale=0.040,
            use_default_offset=False,
        )
    }

    # 6. Observationsのオーバーライド
    # ActorとCriticの両方に同じ観測セットを与えます
    actor_terms = {
        "roll": ObservationTermCfg(func=obs_base_roll),
        "gyro": ObservationTermCfg(func=obs_base_gyro),
        "drive": ObservationTermCfg(func=obs_drive_vel),
    }

    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic": ObservationGroupCfg(
            terms=actor_terms,  # Criticにも同じ情報を与える
            concatenate_terms=True,
            enable_corruption=False,
        )
    }

    # 7. Rewardsのオーバーライド（既存の報酬をクリアして設定）
    cfg.rewards.clear()
    cfg.rewards["survival"] = RewardTermCfg(func=rew_survival, weight=0.0)
    cfg.rewards["upright"] = RewardTermCfg(func=rew_upright, weight=2.0)
    cfg.rewards["odometry"] = RewardTermCfg(func=rew_odometry, weight=-8.0, params={"deadzone": 0.2})
    cfg.rewards["torque"] = RewardTermCfg(func=rew_torque, weight=-0.4)

    # 8. Terminationsのオーバーライド
    cfg.terminations.clear()
    cfg.terminations["time_out"] = TerminationTermCfg(
        func=lambda env: env.episode_length_buf >= env.max_episode_length, 
        time_out=True
    )
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=term_fell_over, 
        params={"limit_angle": math.radians(45.0)}
    )
    cfg.terminations["out_of_bounds"] = TerminationTermCfg(
        func=term_out_of_bounds, 
        params={"limit": 10.5}
    )

    # 9. Events, Commands, Curriculumの初期化/追加
    # cfg.events.clear()
    
    # # 摩擦のランダマイズ (元の値の 0.5 ~ 1.5倍)
    # cfg.events["randomize_friction"] = EventTermCfg(
    #     func=dr.geom_friction,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "operation": "scale",
    #         "ranges": (0.5, 1.5),
    #     }
    # )
    
    # # 質量のランダマイズ (元の値の 0.95 ~ 1.05倍)
    # cfg.events["randomize_mass"] = EventTermCfg(
    #     func=dr.body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "operation": "scale",
    #         "ranges": (0.95, 1.05),
    #     }
    # )

    # # 初期の傾きのランダマイズ (Roll角 ±2.0度)
    # cfg.events["init_tilt"] = EventTermCfg(
    #     func=velocity_mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "pose_range": {
    #             "roll": (-math.radians(2.0), math.radians(2.0)),
    #         },
    #         "velocity_range": {},
    #     }
    # )
    cfg.commands = {}
    cfg.curriculum = {}

    # 10. Playモード（推論・テスト）時の上書き処理
    if play:
        # エピソードを実質無限に
        cfg.episode_length_s = int(1e9)
        
        # ランダマイズと強制リセットを無効化
        cfg.events.pop("randomize", None)
        cfg.events.pop("init_tilt", None)
        
        # テスト時に自由に移動できるよう境界外判定を無効化
        cfg.terminations.pop("out_of_bounds", None)

    return cfg