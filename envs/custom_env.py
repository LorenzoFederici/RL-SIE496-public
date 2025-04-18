import gymnasium as gym

class CustomEnv(gym.Env):
    """
    Custom environment for reinforcement learning.
    This is a simple example of a custom environment that follows the OpenAI Gym interface.
    """

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Max number of steps in the environment
        self.max_steps = 100

        # Define action and observation space
        # They must be gym.spaces objects

        self.action_space = gym.spaces.Discrete(2)  # Example 1: 2 discrete actions
        # self.action_space = gym.spaces.MultiDiscrete([5, 2])  # Example 3: MultiDiscrete action space
        # self.action_space = gym.spaces.MultiBinary(4)  # Example 4: MultiBinary action space
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=float)  # Example 2: 3-dimensional continuous action
        # self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Box(low=0, high=1, shape=(3,), dtype=float)))  # Example 5: Tuple action space
        # self.action_space = gym.spaces.Dict({"action1": gym.spaces.Discrete(2), "action2": gym.spaces.Box(low=0, high=1, shape=(3,), dtype=float)})  # Example 6: Dict action space
        
        self.observation_space = gym.spaces.Discrete(5)  # Example 1: 5 discrete observations
        # self.observation_space = gym.spaces.MultiDiscrete([5, 2])  # Example 3: MultiDiscrete observation space
        # self.observation_space = gym.spaces.MultiBinary(4)  # Example 4: MultiBinary observation space
        # self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=float)  # Example 2: 3-dimensional continuous observation
        # self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Box(low=0, high=1, shape=(3,), dtype=float)))  # Example 5: Tuple observation space
        # self.observation_space = gym.spaces.Dict({"obs1": gym.spaces.Discrete(2), "obs2": gym.spaces.Box(low=0, high=1, shape=(3,), dtype=float)})  # Example 6: Dict observation space


    # Reset the state of the environment to an initial state, and return an initial observation and info
    def reset(self, seed=None, options=None):
        
        self.steps = 0

        observation = self.observation_space.sample() # Example: random initial observation
        
        info = {} # Additional information

        return observation, info


    # Execute one time step within the environment, and return the next observation, reward, truncated, terminated, and info
    def step(self, action):
        
        self.steps += 1
        
        observation = self.observation_space.sample()  # Example: random next observation

        reward = 1.0  # Example: constant reward

        terminated = False
        truncated = False
        if self.steps >= self.max_steps:
            terminated = True

        info = {}  # Additional information

        return observation, reward, terminated, truncated, info


    # Render the environment to the screen (optional)
    def render(self, mode='human'):
        
        pass


    # Close the environment (optional)
    def close(self):
        
        pass