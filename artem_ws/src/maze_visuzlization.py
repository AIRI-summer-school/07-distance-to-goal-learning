import gymnasium as gym
import numpy as np, torch, cv2, imageio, os, base64
from src.env_wrappers import GoalObservationWrapper, TerminateOnSuccessWrapper
from src import ppo_agent
import sys
import time

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


def demo_pointmaze(model_path, out_dir="videos/obs_action",
                   filename="obs_action.mp4",
                   env_id = "PointMaze_UMaze-v3",
                   example_map = None,
                   fps=15, seed=0, deterministic=True):

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
    if not example_map:
        c = 'c'
        example_map = [
            [1, 1, 1, 1, 0],
            [1, c, 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, c, 0, 0, 1],
            [0, 1, 1, 1, 1]
        ]

    env = gym.make(env_id, maze_map=example_map, render_mode="rgb_array")
    env = GoalObservationWrapper(env)
    env = TerminateOnSuccessWrapper(env)

    agent = ppo_agent.PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    agent.ac.load_state_dict(torch.load(model_path, map_location="cpu"))

    frames, obs = [], env.reset(seed=seed)[0]
    done, step = False, 0
    while not done:
        with torch.no_grad():
            act = agent.ac.act(obs, deterministic=deterministic)[0]
        obs, _, term, trunc, _ = env.step(act)
        frame = np.ascontiguousarray(env.render())

        dist = np.hypot(obs[0] - obs[4], obs[1] - obs[5])
        draw_action_arrow(frame, act)
        add_hud(frame, obs, act, step, dist)

        frames.append(frame)
        step += 1
        done = term or trunc
    env.close()

    os.makedirs(out_dir, exist_ok=True)
    # Add timestamp to filename to avoid overwriting
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
