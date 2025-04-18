from typing import Optional
import numpy as np
import gymnasium as gym
import pygame

# This is a simple grid world environment where an agent, 
# starting from a random location, has to reach a target,
# also randomly placed in the grid. The agent can move in
# four directions: right, up, left, and down. The episode
# ends when the agent reaches the target. The reward is 1.
class GridWorldEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode = None, size: int = 5):
        # The size of the square grid
        self.size = size 

        # Max number of steps in the environment
        self.max_steps = int(2*self.size*self.size)

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self.agent_location = np.array([-1, -1], dtype=np.int32)
        self.target_location = np.array([-1, -1], dtype=np.int32)

        # Observation space: the agent and target locations (x,y coordinates)
        self.observation_space = gym.spaces.Box(low = 0, high = self.size - 1, shape=(4,), dtype=int)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)

        # Dictionary maps the abstract actions to the directions on the grid
        self.action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

        # Initialize the render mode
        self.window_size = 512
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
    

    def get_obs(self):

        obs = np.concatenate(
            (self.agent_location, self.target_location),
        dtype=int)
        
        return obs
    

    def get_info(self):
        # Euclidean distance between agent and target
        info = {
            "distance": np.linalg.norm(
                self.agent_location - self.target_location
            )  
        }

        return info
    

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        # Reset the steps counter
        self.steps = 0

        # Choose the agent's location uniformly at random
        self.agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self.target_location = self.agent_location
        while np.array_equal(self.target_location, self.agent_location):
            self.target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self.get_obs()
        info = self.get_info()

        return observation, info
    

    def step(self, action):
        # Convert action to int
        action = int(action)

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self.action_to_direction[action]

        # We use `np.clip` to make sure we don't leave the grid bounds
        self.agent_location = np.clip(
            self.agent_location + direction, 0, self.size - 1
        )

        # An environment is completed if and only if the agent has reached the target
        terminated = np.array_equal(self.agent_location, self.target_location)

        # The episode is truncated if the maximum number of steps is reached
        self.steps += 1
        if self.steps >= self.max_steps and not terminated:
            truncated = True
        else:
            truncated = False
        
        # The reward is -1 for every move, +10 for reaching the target
        reward = -1
        if terminated:
            reward += 10
        
        # The observation is the current agent and target location
        observation = self.get_obs()

        # The info is a dictionary with the distance between the agent and the target
        info = self.get_info()

        return observation, reward, terminated, truncated, info
    

    def render(self):

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()