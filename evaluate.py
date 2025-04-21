import gymnasium as gym
import envs
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Envronment name
env_name = 'GridWorld-v0'
# env_name = 'CartPole-v1'
# env_name = 'Pendulum-v1'
# env_name = 'LunarLander-v3'
# env_name = 'Pusher-v5'

# Create environment
env = gym.make(env_name, render_mode="human")

# Load the trained agent
log = "trained_models/" + env_name + "/"
model = PPO.load(log + "model-ppo")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Reset the environment to generate the first observation
n_steps = 10000
observation, info = env.reset()
for _ in range(n_steps):
    # The trained policy predicts the action, given the observation
    action, _ = model.predict(observation, deterministic=True)

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # Render the environment
    env.render()

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()