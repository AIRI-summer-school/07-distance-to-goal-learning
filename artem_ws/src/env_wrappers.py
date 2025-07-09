import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo, TimeLimit, RecordEpisodeStatistics
import datetime
import os
import cv2, imageio, time, sys, base64



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


class DemonstrateWrapper(gym.Wrapper):
    """
    Wrapper to demonstrate a trained agent in the environment and record a video with HUD overlays.
    Usage:
        env = DemonstrateWrapper(env)
        env.demonstrate(model_path or agent, out_dir, filename, fps, deterministic)
    """

    def demonstrate(self, agent, out_dir="videos/obs_action",
                    filename="obs_action.mp4", fps=15, deterministic=True):
        def draw_action_arrow(img, act):
            h, w = img.shape[:2]
            tail = (int(w * .50), int(h * .90))
            tip  = (int(tail[0] + act[0] * 30),
                    int(tail[1] - act[1] * 30))
            cv2.arrowedLine(img, tail, tip, (255, 0, 0), 2, tipLength=.3)

        def add_hud(img, obs, act, step, dist):
            def panel(origin, size, lines):
                x0, y0 = origin; w, h = size
                cv2.rectangle(img, (x0, y0), (x0+w, y0+h), (255,255,255), -1)
                cv2.addWeighted(img[y0:y0+h, x0:x0+w], 0, img[y0:y0+h, x0:x0+w], .6, 0, img[y0:y0+h, x0:x0+w])
                for i, t in enumerate(lines):
                    font = cv2.FONT_HERSHEY_DUPLEX if i == 0 else cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img, t, (x0+6, y0+18+18*i), font, .55, (0,0,0), 1, cv2.LINE_AA)

            x, y, vx, vy, tx, ty = obs
            panel((5,   5), (190, 82),
                ['OBS', f'pos: [{x:+.2f},{y:+.2f}]',
                        f'vel: [{vx:+.2f},{vy:+.2f}]',
                        f'goal: [{tx:+.2f},{ty:+.2f}]'])
            panel((5,  92), (190, 55),
                ['ACT', f'[{act[0]:+.2f},{act[1]:+.2f}]'])
            panel((200, 5), (200, 82),
                ['INFO', f'step: {step}', f'dist: {dist:.2f}'])


        frames, obs = [], self.reset()[0]
        done, step = False, 0
        while not done:
            with torch.no_grad():
                act = agent.ac.act(obs, deterministic=deterministic)[0]
            obs, _, term, trunc, _ = self.step(act)
            frame = np.ascontiguousarray(self.render())

            dist = np.hypot(obs[0] - obs[4], obs[1] - obs[5])
            draw_action_arrow(frame, act)
            add_hud(frame, obs, act, step, dist)

            frames.append(frame)
            step += 1
            done = term or trunc
        self.close()

        os.makedirs(out_dir, exist_ok=True)
        base, ext = os.path.splitext(filename)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"{base}_{timestamp}{ext}")
        imageio.mimsave(out_path, frames, fps=fps)

        if "IPython" in sys.modules:
            from IPython.display import HTML
            data_url = "data:video/mp4;base64," + base64.b64encode(open(out_path, "rb").read()).decode()
            return HTML(f'<video width="512" controls autoplay loop muted>'
                        f'<source src="{data_url}" type="video/mp4"></video>')
        return out_path

class EnvBuilder:
    """
    Factory-style helper that returns a fully wrapped Gymnasium environment.

    Example
    -------
    >>> builder = EnvBuilder("MazeEnv-v0", max_episode_steps=300, seed=42)
    >>> train_env = builder()          # or builder.build()
    >>> eval_env  = builder(video_folder="videos/")  # record roll-outs only here

    ## Observation
    # | Position | Description                                    | Units   |
    # | -------- | ---------------------------------------------- | ------- |
    # | 0        | `ball_x` — X coordinate of the green ball      | meters  |
    # | 1        | `ball_y` — Y coordinate of the ball            | meters  |
    # | 2        | `vel_x` — ball velocity along X                | m/s     |
    # | 3        | `vel_y` — ball velocity along Y                | m/s     |
    # | 4        | `goal_x` — X coordinate of the goal (red ball) | meters  |
    # | 5        | `goal_y` — Y coordinate of the goal            | meters  |

    ## Action
    # | Index | Component           | Range     | Physical meaning    |
    # | ----- | ------------------- | --------- | ------------------- |
    # | 0     | Fx — force along X  | [-1 … 1]  | `motor_x` in MuJoCo |
    # | 1     | Fy — force along Y  | [-1 … 1]  | `motor_y`           |
    """
    # Easy U-shaped maze
    # Letter 'c' means start and reward place. Each iteration is random
    c = 'c'; simple_map = [
        [1, 1, 1, 1, 1],
        [1, c, 0, 1, 1],
        [1, 1, 0, c, 1],
        [1, 1, 1, 1, 1]
    ]

    def __init__(
        self,
        env_id: str,
        *,
        maze_map: list | np.ndarray = simple_map,
        seed: int | None = None,
        max_episode_steps: int = 200,
        video_folder: str | None = None,
    ):
        self.env_id = env_id
        self.maze_map = maze_map
        self.seed = seed
        self.max_episode_steps = max_episode_steps
        self.video_folder = video_folder

    # Allow the instance to be called like a function → returns an env
    def __call__(self, **overrides):
        return self.build(**overrides)

    def _unique_folder(self, folder: str) -> str:
        if folder and os.path.exists(folder):
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            folder = f"{folder}_{ts}"
        return folder
    
    def get_obs_act_dim(self):
        """
        Returns the observation and action space dimensions after wrapping.
        Useful for defining policy and value networks.
        """
        temp_env = self.build()
        obs_dim = temp_env.observation_space.shape[0]
        act_dim = temp_env.action_space.shape[0]
        temp_env.close()
        return obs_dim, act_dim

    # Explicit constructor in case you prefer builder.build()
    def build(
        self,
        *,
        maze_map: list | np.ndarray | None = None,
        max_episode_steps: int | None = None,
        seed: int | None = None,
        video_folder: str | None = None,
    ):
        # fallback to defaults defined in __init__
        maze_map        = maze_map        if maze_map        is not None else self.maze_map
        max_episode_steps = max_episode_steps if max_episode_steps is not None else self.max_episode_steps
        video_folder    = video_folder    if video_folder    is not None else self.video_folder

        # --- base env ------------------------------------------------------
        env = gym.make(self.env_id, maze_map=maze_map, render_mode="rgb_array")

        # --- optional video recording --------------------------------------
        if video_folder:
            video_folder = self._unique_folder(video_folder)
            env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda eid: True)
        
        # --- wrappers --------------------------------------
        env = GoalObservationWrapper(env)
        env = TerminateOnSuccessWrapper(env)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = RecordEpisodeStatistics(env)
        env = DemonstrateWrapper(env)

        # --- seeding -------------------------------------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            env.action_space.seed(seed)

        return env

    # ── helpers ──────────────────────────────────────────────────────────────
    def make_vec_env(self, num_envs, seed=0, async_mode=True):
        """
        Return a vectorised env (either AsyncVectorEnv or SyncVectorEnv).
        """
        def _thunk(rank):
            return lambda: self.build(seed=seed + rank)
        env_cls = gym.vector.AsyncVectorEnv if async_mode else gym.vector.SyncVectorEnv
        return env_cls([_thunk(i) for i in range(num_envs)])

