import torch
from typing import Optional

from src.luxai_s3.wrappers import LuxAIS3GymEnv

def create_flexible_obs_space(flags, teacher_flags: Optional) -> obs_spaces.BaseObsSpace:
    if teacher_flags is not None and teacher_flags.obs_space != flags.obs_space:
        # Train a student using a different observation space than the teacher
        return obs_spaces.MultiObs({
            "teacher_": teacher_flags.obs_space(**teacher_flags.obs_space_kwargs),
            "student_": flags.obs_space(**flags.obs_space_kwargs)
        })
    else:
        return flags.obs_space(**flags.obs_space_kwargs)

def create_env(flags, device: torch.device, teacher_flags: Optional = None, seed: Optional[int] = None) -> DictEnv:
    if seed is None:
        seed = flags.seed
    envs = []
    for i in range(flags.n_actor_envs):
        env = LuxAIS3GymEnv(
            act_space=flags.act_space(),
            obs_space=create_flexible_obs_space(flags, teacher_flags),
            seed=seed
        )
        reward_space = create_reward_space(flags)
        env = RewardSpaceWrapper(env, reward_space)
        env = env.obs_space.wrap_env(env)
        env = PadFixedShapeEnv(env)
        env = LoggingEnv(env, reward_space)
        envs.append(env)
    env = VecEnv(envs)
    env = PytorchEnv(env, device)
    env = DictEnv(env)
    return env
