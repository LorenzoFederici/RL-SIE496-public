import gymnasium as gym
from stable_baselines3 import DQN, A2C, PPO
import ale_py
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
gym.register_envs(ale_py)

# Envronment name
env_name = 'ALE/Breakout-v5'

# Create environment
env = make_atari_env(env_name, seed=50, env_kwargs={"render_mode": "human"})
env = VecFrameStack(env, n_stack=4) # Stack 4 frames

# Load the trained agent
log = "trained_models/" + env_name + "/"
model = PPO.load(log + "model-ppo")

# Reset the environment to generate the first observation
observation = env.reset()
done = False
while not done:
    # The trained policy predicts the action, given the observation
    action, _ = model.predict(observation, deterministic=True)

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode is done
    observation, reward, done, info = env.step(action)

    # Render the environment
    env.render()
    
env.close()