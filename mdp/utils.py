"""共通ユーティリティ.

クォータニオン演算やロール角計算など、複数モジュールから参照される
低レベルヘルパー関数をまとめるモジュール。
"""

import torch

from mjlab.entity import Entity


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """クォータニオンの逆回転（ワールド座標 → ローカル座標）をベクトルに適用する.

    Args:
        q: クォータニオン [w, x, y, z] 形式 (Shape: [N, 4])
        v: 変換する3次元ベクトル  (Shape: [N, 3])

    Returns:
        変換後の3次元ベクトル     (Shape: [N, 3])
    """
    w = q[..., 0:1]
    u = q[..., 1:4]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v - w * t + torch.cross(u, t, dim=-1)


def compute_roll(asset: Entity) -> torch.Tensor:
    """ワールド座標の重力ベクトルからボディのロール角を計算する.

    重力ベクトルをボディ座標へ変換し、y 成分の arcsin からロール角を求める。

    Args:
        asset: mjlab の Entity オブジェクト（ロボット本体）

    Returns:
        ロール角 [rad]  (Shape: [N])
    """
    gravity_b = quat_apply_inverse(
        asset.data.root_link_quat_w,
        asset.data.gravity_vec_w,
    )
    roll_sin = gravity_b[:, 1]  # y 成分 = sin(roll)
    return torch.asin(torch.clamp(roll_sin, min=-1.0, max=1.0))