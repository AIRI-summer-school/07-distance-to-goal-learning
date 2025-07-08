import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RecordVideo
from dynamical_distance_learning.env_wrappers import GoalObservationWrapper, TerminateOnSuccessWrapper
from dynamical_distance_learning import ppo_agent, distance_models
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os

def main():
    # Stage 2: Train PPO with distance-based reward shaping, updating the distance estimator concurrently
    env_id = 'PointMaze_UMaze-v3'  # sparse reward environment (no dense shaping from env)
    total_timesteps = 50000
    steps_per_iter = 2048
    seed = 0
    torch.manual_seed(seed); np.random.seed(seed)
    # Initialize environment
    gym.register_envs(gymnasium_robotics)
    env = gym.make(env_id)
    env = GoalObservationWrapper(env)
    env = TerminateOnSuccessWrapper(env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # Initialize PPO agent (fresh or could load Stage1 policy)
    agent = ppo_agent.PPOAgent(state_dim=obs_dim, action_dim=act_dim)
    # Load pre-trained distance model from Stage 1 (TD model) for reward shaping
    td_model = distance_models.TDDistanceEstimator(input_dim=obs_dim)
    td_model.load_state_dict(torch.load("models/distance_model_td.pth"))
    td_model.eval()  # set to eval mode for inference
    writer = SummaryWriter(log_dir="runs/stage2")
    global_step = 0
    num_updates = total_timesteps // steps_per_iter
    for update in range(1, num_updates+1):
        traj = agent.collect_trajectory(env, steps_per_iter, use_distance_shaping=True, distance_model=td_model)
        pg_loss, v_loss, ent_loss = agent.update(traj)
        # Concurrently update distance model with new transitions from this batch
        states = traj["states"]; rewards = traj["rewards"]; dones = traj["dones"]
        transitions = []
        for i in range(len(states) - 1):
            if dones[i]:
                # Terminal transition
                success_flag = True if rewards[i] > 0 else False  # in sparse env, reward 1 indicates success
                transitions.append((states[i], states[i], True, success_flag))
            else:
                transitions.append((states[i], states[i+1], False, False))
        # If last state in the batch is terminal, include it as well
        last_idx = len(states) - 1
        if last_idx >= 0 and dones[last_idx]:
            success_flag = True if rewards[last_idx] > 0 else False
            transitions.append((states[last_idx], states[last_idx], True, success_flag))
        td_loss = td_model.train_from_transitions(transitions, epochs=1)
        # Logging metrics
        ep_returns = []
        cum_reward = 0.0
        for r, d in zip(rewards, dones):
            cum_reward += r
            if d:
                ep_returns.append(cum_reward); cum_reward = 0.0
        if ep_returns:
            writer.add_scalar("charts/episodic_return", np.mean(ep_returns), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss, global_step)
        writer.add_scalar("losses/value_loss", v_loss, global_step)
        writer.add_scalar("losses/entropy", ent_loss, global_step)
        writer.add_scalar("losses/distance_loss", td_loss, global_step)
        global_step += len(rewards)
        if update % 10 == 0:
            avg_ret = np.mean(ep_returns) if ep_returns else 0.0
            print(f"Update {update}/{num_updates}: MeanReturn={avg_ret:.2f}")
    # Save Stage 2 agent
    os.makedirs("models", exist_ok=True)
    torch.save(agent.ac.state_dict(), "models/ppo_agent_stage2.pth")
    # Record video of Stage 2 agent in the sparse-reward environment
    eval_env = gym.make(env_id)
    eval_env = GoalObservationWrapper(eval_env)
    eval_env = TerminateOnSuccessWrapper(eval_env)
    video_env = RecordVideo(eval_env, video_folder="videos/stage2", episode_trigger=lambda eid: True)
    obs, _ = video_env.reset()
    done = False
    while not done:
        action, _, _ = agent.ac.act(obs)
        obs, _, terminated, truncated, info = video_env.step(action)
        done = terminated or truncated
    video_env.close()
    env.close(); eval_env.close()

if __name__ == "__main__":
    main()
