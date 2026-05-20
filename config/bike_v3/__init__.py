from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.bike_task.rl import StandingOnPolicyRunner

from .env_cfgs import (
  bike_v3_env_cfg,
)
from .rl_cfg import bike_v3_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Standing-bike-v3",
  env_cfg=bike_v3_env_cfg(),
  play_env_cfg=bike_v3_env_cfg(play=True),
  rl_cfg=bike_v3_ppo_runner_cfg(),
  runner_cls=StandingOnPolicyRunner,
)

# register_mjlab_task(
#   task_id="Mjlab-Standing-Flat-Unitree-G1",
#   env_cfg=unitree_g1_flat_env_cfg(),
#   play_env_cfg=unitree_g1_flat_env_cfg(play=True),
#   rl_cfg=bike_v3_ppo_runner_cfg(),
#   runner_cls=StandingOnPolicyRunner,
# )
