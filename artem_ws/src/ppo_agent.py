import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import wandb
from tqdm import trange

class ActorCritic(nn.Module):
    """
    A combined actor-critic network that outputs both action distribution (for policy)
    and state-value (for critic) given an input state.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Simple MLP network
        hidden_dim = 64
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # Separate linear layer for value function
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Learnable log standard deviation for Gaussian policy (diagonal covariance)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        """
        Given state input (torch tensor of shape (N, state_dim)),
        return a tuple (dist, value) where dist is a Normal distribution for actions,
        and value is the state-value prediction.
        """
        mean = self.policy_net(state)
        # Create diagonal Gaussian distribution for continuous actions
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        value = self.value_net(state).squeeze(-1)  # shape (N,)
        return dist, value

    def act(self, state, deterministic=False):
        """
        Select an action from the policy given a single state (numpy) input.
        If deterministic=True, returns the mean action (no sampling).
        If deterministic=False (default), samples from the policy distribution.
        Returns action (numpy array), log probability, and value.
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():                     # <-- инференс без автограда
            dist, value = self.forward(state_t)
            action = dist.mean if deterministic else dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()

    def evaluate_actions(self, states, actions):
        """
        Evaluate log probabilities, values and entropy for given states and actions.
        Used for computing PPO losses.
        """
        dist, values = self.forward(states)
        log_probs = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return log_probs, values, entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lambd=0.95, clip_coef=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, log_to_wandb: bool = True, wandb_name = "runs/stage1"):
        """
        PPO Agent containing policy network and optimizer, with training methods.
        """
        self.gamma = gamma
        self.lambd = lambd
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.log_to_wandb = log_to_wandb
        # Create actor-critic network and optimizer
        self.ac = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

        if log_to_wandb:
            self.run = wandb.init(
                project="ppo_rl", name=wandb_name,
            )
            
    def collect_trajectory(self, env, timestep_limit, use_distance_shaping=False, distance_model=None):
        """
        Collect a trajectory of experiences by running the policy in the environment.
        Collects up to `timestep_limit` steps, possibly spanning multiple episodes.
        Optionally uses a distance_model to shape rewards.
        Returns a dictionary with arrays: states, actions, rewards, dones, log_probs, values, next_value.
        """
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []
        # Reset environment for starting state
        state, _ = env.reset()
        for t in range(timestep_limit):
            action, log_prob, value = self.ac.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # If using distance-based reward shaping, adjust reward:contentReference[oaicite:0]{index=0}
            if use_distance_shaping and distance_model is not None:
                # Compute shaping reward using learned distance as potential
                current_distance = distance_model.predict(state)
                next_distance = 0.0
                if not terminated:
                    next_distance = distance_model.predict(next_state)
                # Potential-based shaping: reward += (current_distance - next_distance)
                # This gives a positive reward for decreasing the distance (moving closer to goal).
                reward += (current_distance - next_distance)
            # Save transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)
            # Move to next state
            state = next_state
            if done:
                # Episode ended, reset environment to start a new episode if more timesteps remain
                state, _ = env.reset()
                # Note: We continue collecting; 'done' flags break advantage calculation properly
        # After loop, collect final value for bootstrap (for advantage calc)
        if done:
            next_value = 0.0
        else:
            # If time limit reached without termination, estimate value of last state for bootstrap
            next_value = self.ac.forward(torch.from_numpy(state).float().unsqueeze(0))[1].item()
        # Convert lists to numpy arrays
        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
            "log_probs": np.array(log_probs, dtype=np.float32),
            "values": np.array(values, dtype=np.float32),
            "next_value": np.array(next_value, dtype=np.float32)
        }

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE) for a trajectory.
        Returns advantage and returns (value targets).
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(T)):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_adv = delta  # if done, no bootstrapping from next state
            else:
                delta = rewards[t] + self.gamma * (values[t+1] if t+1 < T else next_value) - values[t]
                last_adv = delta + self.gamma * self.lambd * last_adv
            advantages[t] = last_adv
        returns = advantages + values[:T]
        return advantages, returns

    def update(self, trajectory, batch_size=64, epochs=4):
        """
        Perform PPO updates given a trajectory (potentially containing multiple episodes).
        Uses multiple epochs and minibatches for optimization.
        """
        states = torch.tensor(trajectory["states"], dtype=torch.float32)
        actions = torch.tensor(trajectory["actions"], dtype=torch.float32)
        old_log_probs = torch.tensor(trajectory["log_probs"], dtype=torch.float32)
        # Concatenate values with the bootstrap value for advantage calculation
        values_all = np.concatenate([trajectory["values"], [trajectory["next_value"]]])
        advantages, returns = self.compute_gae(trajectory["rewards"], values_all, trajectory["dones"], trajectory["next_value"])
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # PPO multiple epoch update
        dataset_size = states.size(0)
        for epoch in range(epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                # Evaluate current policy on minibatch
                new_log_probs, values_pred, entropy = self.ac.evaluate_actions(mb_states, mb_actions)
                # Policy loss (clipped surrogate objective)
                ratios = torch.exp(new_log_probs - mb_old_log_probs)
                pg_loss1 = -mb_adv * ratios
                pg_loss2 = -mb_adv * torch.clamp(ratios, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                pg_loss = torch.mean(torch.min(pg_loss1, pg_loss2))
                # Value function loss
                value_loss = torch.mean((values_pred - mb_returns)**2)
                # Entropy bonus (for exploration)
                entropy_loss = torch.mean(entropy)
                # Total loss
                loss = pg_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss
                # Optimize policy and value network
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()
        # Return losses for logging
        return pg_loss.item(), value_loss.item(), entropy_loss.item()


    def train_ppo(
        self,
        env,
        *,
        num_episodes: int,
        max_episode_steps: int,
    ):
        for episode in trange(1, num_episodes + 1, desc="Training", ncols=100):
            traj = self.collect_trajectory(env, max_episode_steps)
            pg_loss, v_loss, ent_loss = self.update(traj)

            rewards, dones = traj["rewards"], traj["dones"]
            ep_returns = np.add.reduceat(rewards, np.flatnonzero(np.r_[True, dones[:-1]]))
            if ep_returns.size:
                wandb.log({"charts/episodic_return": ep_returns.mean()},step=episode)
            wandb.log( {
                    "losses/policy_loss": pg_loss,
                    "losses/value_loss": v_loss,
                    "losses/entropy": ent_loss,
                },
                step=episode,
            )


    def evaluate_ppo(
        self,
        env,
        *,
        num_episodes: int = 100,
        max_episode_steps: int | None = 2048,
        deterministic: bool = True,
    ): 
        trajectories = []
        returns = []
        with torch.no_grad():
            for ep in trange(num_episodes, desc="Evaluating", ncols=100):
                state, _ = env.reset()
                ep_return = 0.0
                episode_traj = []
                for _ in range(max_episode_steps):
                    ball_x, ball_y = state[0], state[1]
                    episode_traj.append((ball_x, ball_y))

                    action, *_ = self.ac.act(state, deterministic=deterministic)
                    state, reward, terminated, truncated, _ = env.step(action)
                    ep_return += reward
                    if terminated or truncated:
                        break
                trajectories.append(episode_traj)
                returns.append(ep_return)

        if self.log_to_wandb and wandb.run is not None:
            wandb.log({"eval/episodic_return": np.mean(returns)})

        print(f"Over {num_episodes} eval episodes, {np.count_nonzero(returns)} were successful ({100*np.mean(returns):.1f}%)")
        return trajectories
    
    