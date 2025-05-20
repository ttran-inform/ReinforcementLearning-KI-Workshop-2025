from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

env = make_vec_env('LunarLander-v2', n_envs=16)

model = PPO(
    policy='MlpPolicy',
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1
)

model.learn(total_timesteps=1024*100)
model_name = "ppo-LunarLander-v2"
# model.save(model_name)

eval_env = gym.make("LunarLander-v2", render_mode='rgb_array')
eval_env = RecordVideo(
    env=eval_env,
    video_folder="LunarLander/videos",
    episode_trigger=lambda episode_id: episode_id < 5,
    name_prefix="ppo_eval"
)
eval_env = Monitor(eval_env)

mean_reward, std_reward = evaluate_policy(
    model, eval_env,
    n_eval_episodes=5,
    deterministic=True
)

print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

eval_env.close()
