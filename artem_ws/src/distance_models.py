import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np

class BaseDistanceEstimator(nn.Module):
    """
    Base class for distance estimators (dynamical distances). It holds a neural network
    that predicts the distance (expected number of time steps) to reach the goal from a given state:contentReference[oaicite:1]{index=1}.
    """
    def __init__(self, input_dim, lr=1e-3):
        super().__init__()
        # Simple MLP to predict a single scalar distance
        hidden_dim = 64
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    def forward(self, x):
        # Forward pass: returns distance prediction as a tensor of shape (N,1)
        return self.model(x)
    def predict(self, state):
        """
        Predict distance for a single state (numpy array). Returns a float.
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        dist_pred = self.forward(state_t)
        return dist_pred.item()

    def plot_distance_heatmap(self, source_point=(0.0, 0.0), grid_size=5000, xlim=(-2, 2), ylim=(-2, 2)):
        """
        Plots a heatmap of predicted distances from a fixed source point to a grid of target points.
        """
        x_src, y_src = source_point

        # Create a 2D grid of target points
        x_vals = np.linspace(*xlim, grid_size)
        y_vals = np.linspace(*ylim, grid_size)
        xx, yy = np.meshgrid(x_vals, y_vals)
        target_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

        # Build model input: [x_src, y_src, x_tgt, y_tgt]
        input_states = np.hstack([
            np.full((target_points.shape[0], 2), [x_src, y_src]),
            target_points
        ])

        # Predict distances
        with torch.no_grad():
            inputs = torch.tensor(input_states, dtype=torch.float32)
            preds = self.forward(inputs).squeeze().numpy()

        # Reshape predictions to grid for plotting
        heatmap = preds.reshape(grid_size, grid_size)

        # Plot
        plt.figure(figsize=(6, 5))
        plt.imshow(
            heatmap,
            origin='lower',
            extent=(*xlim, *ylim),
            cmap='viridis',
            aspect='auto'
        )
        plt.colorbar(label="Predicted Distance")
        plt.scatter(*source_point, color='red', label='Source', s=60, edgecolors='black')
        plt.xlabel("Target X")
        plt.ylabel("Target Y")
        plt.title(f"Distance Heatmap from Source ({x_src:.1f}, {y_src:.1f})")
        plt.legend()
        plt.tight_layout()
        plt.show()

class SupervisedDistanceEstimator(BaseDistanceEstimator):
    """
    Distance estimator trained with supervised labels (true distances from states to goal).
    """
    def train_from_data(self, states, distances, epochs=50, batch_size=64):
        """
        Train the distance model using supervised data.
        states: numpy array of shape (N, state_dim), distances: numpy array of shape (N,)
        """
        dataset_size = states.shape[0]
        states_t = torch.tensor(states, dtype=torch.float32)
        targets_t = torch.tensor(distances, dtype=torch.float32).unsqueeze(1)
        for epoch in trange(epochs, desc="Training distance model", ncols=100):

            # Shuffle indices for mini-batch training
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                batch_states = states_t[batch_idx]
                batch_targets = targets_t[batch_idx]
                # Forward pass
                preds = self.forward(batch_states)
                loss = self.loss_fn(preds, batch_targets)
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        # Final training loss on full dataset
        final_loss = self.loss_fn(self.forward(states_t), targets_t).item()
        return final_loss

class TDDistanceEstimator(BaseDistanceEstimator):
    """
    Distance estimator trained via temporal-difference (Bellman updates) on transitions.
    """
    def train_from_transitions(self, transitions, epochs=50, batch_size=64):
        """
        Train the distance model using temporal-difference updates.
        transitions: list of tuples (state, next_state, done, success) for each transition.
                     'done' indicates episode ended after this transition,
                     'success' indicates termination was a successful goal reach.
        """
        # Prepare tensors for states and next_states
        states = torch.tensor([t[0] for t in transitions], dtype=torch.float32)
        next_states = torch.tensor([t[1] for t in transitions], dtype=torch.float32)
        dones = [t[2] for t in transitions]
        successes = [t[3] for t in transitions]
        N = states.shape[0]
        for epoch in trange(epochs, desc="Training distance model", ncols=100):

            order = torch.randperm(N)
            for start in range(0, N, batch_size):
                end = start + batch_size
                batch_idx = order[start:end].tolist()
                batch_states = states[batch_idx]
                batch_next_states = next_states[batch_idx]
                batch_dones = [dones[i] for i in batch_idx]
                batch_successes = [successes[i] for i in batch_idx]
                # Compute targets for each transition in the batch
                with torch.no_grad():
                    next_dists = self.forward(batch_next_states).squeeze(1)
                targets = []
                for i, done in enumerate(batch_dones):
                    if done:
                        if batch_successes[i]:
                            # Successful terminal transition: distance to goal = 1 (one step remaining)
                            targets.append(1.0)
                        else:
                            # Terminal due to failure (e.g., timeout): skip learning this transition
                            targets.append(None)
                    else:
                        # Non-terminal: target = 1 + predicted distance of next state
                        targets.append(1.0 + next_dists[i].item())
                # Filter out transitions with undefined targets (failures)
                valid_indices = [idx for idx, t in enumerate(targets) if t is not None]
                if len(valid_indices) == 0:
                    continue
                batch_states_valid = batch_states[valid_indices]
                batch_targets = torch.tensor([targets[idx] for idx in valid_indices], dtype=torch.float32).unsqueeze(1)
                # TD loss (MSE between current prediction and target)
                preds = self.forward(batch_states_valid)
                loss = self.loss_fn(preds, batch_targets)
                # Backpropagate
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        # Compute final loss over all valid transitions for reporting
        with torch.no_grad():
            all_next = self.forward(next_states).squeeze(1)
        all_targets = []
        valid_mask = []
        for j, (s, ns, done, succ) in enumerate(transitions):
            if done:
                if succ:
                    all_targets.append(1.0)
                    valid_mask.append(True)
                else:
                    # failure terminal, exclude from evaluation
                    valid_mask.append(False)
                    all_targets.append(0.0)
            else:
                all_targets.append(1.0 + all_next[j].item())
                valid_mask.append(True)
        valid_states = states[[i for i, m in enumerate(valid_mask) if m]]
        valid_targets = torch.tensor([t for t, m in zip(all_targets, valid_mask) if m], dtype=torch.float32).unsqueeze(1)
        final_loss = self.loss_fn(self.forward(valid_states), valid_targets).item()
        return final_loss


