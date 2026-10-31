from env import make_env
import gymnasium as gym

SEED = 69420

# On cree 28 environements 
envs = gym.vector.SyncVectorEnv(
        [make_env("ALE/Breakout-v5", i, True, "test") for i in range(28)]
    )

next_obs, _ = envs.reset(seed=SEED)

for iteration in range(100) :
    action = envs.action_space.sample() # On prend 28 actions aléatoires parmis celles possibles
    next_obs, reward, terminations, truncations, infos = envs.step(action)

envs.close()