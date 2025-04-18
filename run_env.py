import gymnasium as gym
import envs
import ale_py
gym.register_envs(ale_py)

# Envronment name
# env_name = 'CartPole-v1'
# env_name = 'Pendulum-v1'
# env_name = 'LunarLander-v3'
# env_name = 'Pusher-v5'
# env_name = 'ALE/Breakout-v5'
env_name = 'GridWorld-v0'

# Create environment
env = gym.make(env_name, render_mode="human")

# Reset the environment to generate the first observation
n_steps = 10000
observation, info = env.reset()
for _ in range(n_steps):
    # sampling from the action space. This is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # Render the environment
    env.render()

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()