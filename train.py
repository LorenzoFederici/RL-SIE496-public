import gymnasium as gym
import envs
from stable_baselines3 import DQN, A2C, PPO
import torch as th

# Envronment name
env_name = 'GridWorld-v0'
# env_name = 'CartPole-v1'
# env_name = 'Pendulum-v1'
# env_name = 'LunarLander-v3'
# env_name = 'Pusher-v5'

# Create environment
env = gym.make(env_name)

# Policy
policy_name = "MlpPolicy"

# Custom actor (pi) and value function (vf) networks (optional)
policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64])
)

# Instantiate the agent
log = "trained_models/" + env_name + "/"
model = PPO(
        policy = policy_name,
        policy_kwargs=policy_kwargs,
        learning_rate = 2.5e-4,
        n_steps = 5000,
        batch_size = 500,
        n_epochs = 30,
        gamma = 0.99,
        gae_lambda = 0.95,
        clip_range = 0.1,
        env = env, 
        tensorboard_log = log,
        verbose = 1)

# Save random agent
model.save(log + "model-random")

# Train the agent and display a progress bar
n_train_steps = int(1e6)
model.learn(total_timesteps=n_train_steps, 
            progress_bar=True)

# Save the agent
model.save(log + "model-ppo")
