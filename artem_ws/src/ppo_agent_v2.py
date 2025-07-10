import gymnasium as gym
import gymnasium_robotics  # If not used elsewhere, you may remove this import.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import wandb
import time
import random


# ──────────────────────────────────────────────────────────────────────────────
#  Actor‑Critic network (unchanged apart from "input_dim" rename for clarity)
# ──────────────────────────────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    """Simple 2‑layer MLP actor‑critic."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()

        def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)
            return layer

        # Critic ▸ V(s)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        # Actor ▸ π(a|s)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    # ── helpers ──────────────────────────────────────────────────────────────
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        mean = self.actor_mean(x)
        std  = torch.exp(self.actor_logstd).expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        value    = self.critic(x).squeeze(-1)
        return action, log_prob, entropy, value


# ──────────────────────────────────────────────────────────────────────────────
#  PPO agent with optional distance‑augmented state
# ──────────────────────────────────────────────────────────────────────────────
class PPOAgent:
    """Proximal Policy Optimisation with optional distance feature in the state."""


    def save_model(self, path=None):
        model_path = path or f"runs/ppo_{time.time()}.cleanrl_model"
        torch.save(self.agent.state_dict(), model_path)
        print(f"Model saved to {model_path}")


    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        # PPO hyper‑params ----------------------------------------------------
        learning_rate: float = 3e-4,
        anneal_lr: bool = True,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        update_epochs: int = 10,
        norm_adv: bool = True,
        clip_coef: float = 0.2,
        clip_vloss: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float | None = None,
        # Training loop ------------------------------------------------------
        num_envs: int = 1,
        num_steps: int = 2048,
        num_minibatches: int = 32,
        total_timesteps: int = 300_000,
        # Distance feature ---------------------------------------------------
        include_distance_state: bool = False,
        distance_model: nn.Module | None = None,
        # Logging ------------------------------------------------------------
        log_to_wandb: bool = True,
        wandb_project: str = "ppo_airi",
        wandb_run_name: str | None = None,
    ) -> None:
        # Store basic params --------------------------------------------------
        self.include_distance_state = include_distance_state
        self.distance_model         = distance_model  # can be None if flag False
        if self.include_distance_state and self.distance_model is None:
            raise ValueError("include_distance_state=True requires a distance_model instance.")

        # Input dimension ↑ by 1 when we append distance prediction
        self.obs_dim        = state_dim + (1 if self.include_distance_state else 0)
        self.action_dim     = action_dim

        # PPO hyper‑parameters ----------------------------------------------
        self.gamma          = gamma
        self.gae_lambda     = gae_lambda
        self.update_epochs  = update_epochs
        self.norm_adv       = norm_adv
        self.clip_coef      = clip_coef
        self.clip_vloss     = clip_vloss
        self.ent_coef       = ent_coef
        self.vf_coef        = vf_coef
        self.max_grad_norm  = max_grad_norm
        self.target_kl      = target_kl
        self.learning_rate  = learning_rate
        self.anneal_lr      = anneal_lr

        # Rollout parameters --------------------------------------------------
        self.num_envs       = num_envs
        self.num_steps      = num_steps
        self.batch_size     = self.num_envs * self.num_steps
        self.num_minibatches = num_minibatches
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.total_timesteps = total_timesteps
        self.num_iterations  = self.total_timesteps // self.batch_size

        # Networks & optimiser ----------------------------------------------
        self.agent     = ActorCritic(self.obs_dim, self.action_dim)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate)

        # Logging ------------------------------------------------------------
        self.log_to_wandb = log_to_wandb
        if log_to_wandb:
            self.run_name = wandb_run_name or f"ppo_{int(time.time())}"
            self.run = wandb.init(
                project=wandb_project,
                name=self.run_name,
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
            )
            self.writer = SummaryWriter(f"runs/{self.run_name}")
        else:
            self.writer = SummaryWriter(log_dir="/tmp/ppo_logs")

    # ──────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _augment_obs(self, obs_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Append learned distance as extra feature when enabled."""
        if not self.include_distance_state:
            return obs_tensor.to(device)

        # obs_tensor shape (..., 6) → we need (pos_x,pos_y,go_x,go_y)
        pos  = obs_tensor[..., :2]
        goal = obs_tensor[..., 4:6]
        inp  = torch.cat([pos, goal], dim=-1).to(device)  # (...,4)
        self.distance_model.to(device).eval()
        dist_pred = self.distance_model(inp).squeeze(-1)  # (...,)
        # ensure same ndim ⇒ unsqueeze(‑1)
        dist_pred = dist_pred.unsqueeze(-1)
        # concat along last dim
        return torch.cat([obs_tensor.to(device), dist_pred], dim=-1)

    # ──────────────────────────────────────────────────────────────────────
    #  Training loop (single‑env for clarity)
    # ──────────────────────────────────────────────────────────────────────
    def train_ppo(
        self,
        envs: gym.vector.VectorEnv,
        *,
        seed: int = 1,
        torch_deterministic: bool = True,
        use_distance_shaping: bool = False,
        verbose = True
    ) -> None:
        """Main PPO learning loop (single / vector env)."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent.to(device)

        # ▸ rollout buffers --------------------------------------------------
        obs_buf     = torch.zeros((self.num_steps, self.num_envs, self.obs_dim), device=device)
        actions_buf = torch.zeros((self.num_steps, self.num_envs, self.action_dim), device=device)
        logp_buf    = torch.zeros((self.num_steps, self.num_envs), device=device)
        rews_buf    = torch.zeros((self.num_steps, self.num_envs), device=device)
        dones_buf   = torch.zeros((self.num_steps, self.num_envs), device=device)
        values_buf  = torch.zeros((self.num_steps, self.num_envs), device=device)

        # ▸ initial observation ----------------------------------------------
        next_obs_np, _ = envs.reset(seed=seed)            # (N, 6)
        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32)
        next_obs = self._augment_obs(next_obs, device)    # maybe add distance
        next_done = torch.zeros(self.num_envs, device=device)

        global_step = 0
        start_time = time.time()

        for iter_idx in range(1, self.num_iterations + 1):
            # optional LR annealing
            if self.anneal_lr:
                frac = 1.0 - (iter_idx - 1.0) / self.num_iterations
                self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

            for step in range(self.num_steps):
                global_step += self.num_envs
                obs_buf[step]   = next_obs
                dones_buf[step] = next_done

                # ── policy inference ──────────────────────────────────────
                with torch.no_grad():
                    action, logp, _, value = self.agent.get_action_and_value(next_obs)
                    values_buf[step] = value.flatten()
                actions_buf[step] = action
                logp_buf[step]    = logp

                # ── env step ─────────────────────────────────────────────
                next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

                # optional distance‑based reward shaping ------------------
                if use_distance_shaping and self.distance_model is not None:
                    self.distance_model.to(device).eval()
                    with torch.no_grad():
                        cur_pos  = next_obs[..., :2]
                        goal_pos = next_obs[..., 4:6]
                        next_pos = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)[..., :2]
                        d_curr = self.distance_model(torch.cat([cur_pos, goal_pos], -1)).squeeze(-1)
                        d_next = self.distance_model(torch.cat([next_pos, goal_pos], -1)).squeeze(-1)
                    reward = reward + self.gamma * (d_curr.cpu().numpy() - d_next.cpu().numpy())

                # store reward & done
                rews_buf[step]  = torch.as_tensor(reward, device=device)
                next_done        = torch.as_tensor(np.logical_or(terminations, truncations), dtype=torch.float32, device=device)

                # augment next_obs again
                next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32)
                next_obs = self._augment_obs(next_obs, device)

                # Log episodic returns if provided by env ------------------
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            ep_r, ep_l = info["episode"]["r"], info["episode"]["l"]
                            if verbose:
                                print(f"global_step={global_step}, episodic_return={ep_r}")
                            self.writer.add_scalar("charts/episodic_return", ep_r, global_step)
                            self.writer.add_scalar("charts/episodic_length",  ep_l, global_step)

            # ── GAE + returns ───────────────────────────────────────────────
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(self.num_envs)
                advantages = torch.zeros_like(rews_buf, device=device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues      = next_value
                    else:
                        nextnonterminal = 1.0 - dones_buf[t + 1]
                        nextvalues      = values_buf[t + 1]
                    delta = rews_buf[t] + self.gamma * nextvalues * nextnonterminal - values_buf[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values_buf

            # ── flatten for optimisation ----------------------------------
            b_obs       = obs_buf.reshape(self.batch_size, self.obs_dim)
            b_actions   = actions_buf.reshape(self.batch_size, self.action_dim)
            b_logp      = logp_buf.reshape(self.batch_size)
            b_adv       = advantages.reshape(self.batch_size)
            b_ret       = returns.reshape(self.batch_size)
            b_values    = values_buf.reshape(self.batch_size)

            # ── PPO mini‑batch updates ------------------------------------
            b_inds   = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, new_logp, entropy, new_value = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    log_ratio = new_logp - b_logp[mb_inds]
                    ratio     = log_ratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-log_ratio).mean()
                        approx_kl     = ((ratio - 1) - log_ratio).mean()
                        clipfracs.append(((ratio - 1.0).abs() > self.clip_coef).float().mean().item())

                    mb_adv_v = b_adv[mb_inds]
                    if self.norm_adv:
                        mb_adv_v = (mb_adv_v - mb_adv_v.mean()) / (mb_adv_v.std() + 1e-8)

                    # surrogate loss
                    pg_loss1 = -mb_adv_v * ratio
                    pg_loss2 = -mb_adv_v * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                    # value loss
                    new_value = new_value.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (new_value - b_ret[mb_inds]) ** 2
                        v_clipped        = b_values[mb_inds] + torch.clamp(new_value - b_values[mb_inds], -self.clip_coef, self.clip_coef)
                        v_loss_clipped   = (v_clipped - b_ret[mb_inds]) ** 2
                        v_loss           = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * (new_value - b_ret[mb_inds]).pow(2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                # early stop on KL ------------------------------------------------
                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

            # ── logging -----------------------------------------------------
            if self.log_to_wandb:
                self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                self.writer.add_scalar("losses/value_loss",  v_loss.item(), global_step)
                self.writer.add_scalar("losses/entropy",     entropy_loss.item(), global_step)
                self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # ──────────────────────────────────────────────────────────────────────
    #  Evaluation (distance‑augmented)
    # ──────────────────────────────────────────────────────────────────────
    def evaluate_ppo(
        self,
        env: gym.Env,
        *,
        num_episodes: int = 100,
        max_episode_steps: int | None = 170,
        deterministic: bool = True,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent.to(device).eval()

        trajectories   = []
        ep_returns     = []
        success_count  = 0

        for _ in trange(num_episodes, desc="Evaluating", ncols=100):
            obs_np, _ = env.reset()
            obs = torch.as_tensor(obs_np, dtype=torch.float32)
            obs = self._augment_obs(obs, device)

            episode_traj = []
            for step in range(max_episode_steps):
                ball_x, ball_y = obs.cpu().numpy()[:2]
                episode_traj.append((ball_x, ball_y))

                with torch.no_grad():
                    action, *_ = self.agent.get_action_and_value(obs.unsqueeze(0), None)
                next_obs_np, _, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
                obs = torch.as_tensor(next_obs_np, dtype=torch.float32)
                obs = self._augment_obs(obs, device)

                if info.get("success", False):
                    success_count += 1
                    ep_returns.append(max_episode_steps - step)   # reward
                    break
                if terminated or truncated:
                    break

            trajectories.append(episode_traj)

        # ---------- statistics ----------
        success_rate = success_count / num_episodes
        avg_return   = np.mean(ep_returns) if ep_returns else 0.0
        std_return   = np.std(ep_returns, ddof=1) if len(ep_returns) > 1 else 0.0

        print(f"Success rate: {success_rate:.2%}, "
            f"avg_return: {avg_return:.2f} ± {std_return:.2f}")

        return trajectories

