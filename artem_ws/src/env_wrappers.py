import gymnasium as gym
import numpy as np

class GoalObservationWrapper(gym.ObservationWrapper):
    """
    Observation wrapper for goal-based environments. Converts the observation (dict with 'observation' and 'desired_goal')
    into a single numpy array by concatenating the agent's state and the goal. 
    (The 'achieved_goal' is omitted as it duplicates the agent position.)
    """
    def __init__(self, env):
        super().__init__(env)
        # Determine original observation space shape
        obs_shape = env.observation_space['observation'].shape
        goal_shape = env.observation_space['desired_goal'].shape
        assert len(obs_shape) == 1 and len(goal_shape) == 1
        new_dim = obs_shape[0] + goal_shape[0]
        # Define new observation space (continuous Box with broad bounds)
        low = -np.inf * np.ones(new_dim, dtype=np.float32)
        high = np.inf * np.ones(new_dim, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(new_dim,), dtype=np.float32)
    def observation(self, obs):
        # Concatenate the original state observation and desired goal into one array
        return np.concatenate([obs['observation'], obs['desired_goal']], dtype=np.float32)

class TerminateOnSuccessWrapper(gym.Wrapper):
    """
    Wrapper to terminate the episode when a 'success' info flag is True.
    Some Maze environments can be continuing tasks; this wrapper forces episode termination upon success.
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get('success', False):
            terminated = True  # end episode when success achieved
        return obs, reward, terminated, truncated, info
