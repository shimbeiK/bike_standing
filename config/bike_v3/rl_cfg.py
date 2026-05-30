"""RL configuration for bike V3 bicycle task."""

import math
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

def bike_v3_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for bike V3 bicycle task."""
  
  # log_std_init: -0.3 を標準偏差の初期値(init_std)に変換
  init_std_val = math.exp(0)

  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 64),  # net_arch: [128, 64]
      activation="elu",      # sb3のデフォルトはtanh等ですが、mjlabの標準に合わせeluとしています
      obs_normalization=False,
      distribution_cfg={     # 初期行動の分散を定義
        "class_name": "GaussianDistribution",
        "init_std": init_std_val,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(128, 64),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,         # clip_range: 0.3
      entropy_coef=0.0,      # コメント(0.01)と値(0.05)がありましたが0.05を仮設定
      num_learning_epochs=10, # n_epochs: 10 に相当
      num_mini_batches=4,     # batch_sizeとn_stepsから適宜分割数を設定
      learning_rate=3e-4,     # learning_rate: 3e-4
      schedule="adaptive",
      gamma=0.99,            # gamma: 0.995
      lam=0.95,               # gae_lambda: 0.95
      desired_kl=0.01,
      max_grad_norm=1.0,      # max_grad_norm: 0.3
    ),
    experiment_name="bike_v3",
    save_interval=100,
    num_steps_per_env=24,   # n_steps: 8192
    max_iterations=1_500,
  )