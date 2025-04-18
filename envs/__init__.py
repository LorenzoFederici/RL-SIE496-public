from gymnasium.envs.registration import register

# Register the custom environment
register(
    id="CustomEnv-v0",
    entry_point="envs.custom_env:CustomEnv",
)

# Register the GridWorld environment
register(
    id="GridWorld-v0",
    entry_point="envs.gridworld_env:GridWorldEnv",
    max_episode_steps=300,
)