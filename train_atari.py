import gymnasium as gym
import envs
from stable_baselines3 import DQN, A2C, PPO
import torch as th
import ale_py
from utils.custom_policies import CustomCNN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from utils.schedules import linear_schedule
gym.register_envs(ale_py)

# Envronment name
env_name = 'ALE/Breakout-v5'

# Create environment
env = make_atari_env(env_name, n_envs=8)
env = VecFrameStack(env, n_stack=4) # Stack 4 frames

# For Atari games, use CnnPolicy as the base policy
policy_name = "CnnPolicy"

# # Custom feature extractor (optional)
# policy_kwargs = dict(
#         features_extractor_class=CustomCNN,
#         features_extractor_kwargs=dict(features_dim=128),
#         activation_fn=th.nn.ReLU,
#         net_arch=dict(pi=[256], vf=[256]),
# )
policy_kwargs = None

# Instantiate the agent
log = "trained_models/" + env_name + "/"
model = PPO(
        policy = policy_name,
        policy_kwargs=policy_kwargs,
        learning_rate = linear_schedule(2.5e-4),
        n_steps = 128,
        batch_size = 256,
        n_epochs = 4,
        gamma = 0.99,
        gae_lambda = 0.95,
        clip_range = linear_schedule(0.1),
        ent_coef = 0.01,
        env = env, 
        tensorboard_log = log,
        verbose = 1)

# Save random agent
model.save(log + "model-random")

# Train the agent and display a progress bar
n_train_steps = int(10e6)
model.learn(total_timesteps=n_train_steps, 
            progress_bar=True)

# Save the agent
model.save(log + "model-ppo")
