import gymnasium as gym
import ale_py

from atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


# Fortement inspire par https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py#L89


def make_env(env_id, idx, capture_video, run_name):
    # Pour info "ALE/Breakout-v5" a deja un frameskip de 4
    # et repeat_action_probability=0.25 on fixe just frameskip
    # pour les autres jeux et on garde le repeat par default

    def wrapper() : # On est obliger de creer une fonction et de la return sinon ça fait une erreur

        frames_to_skip = 4
        
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", frameskip=frames_to_skip)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, frameskip=frames_to_skip)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        return env

    return wrapper