import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb
import numpy as np
import mujoco 

from tqdm import trange

class BaseDistanceEstimator(nn.Module):
    """
    Lightweight MLP that maps R^input_dim → R (distance).
    """
    def __init__(self, input_dim, lr=1e-3, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.start_point=(0., 0.)
        self.xlim=(-2, 2)
        self.ylim=(-2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def plot_distance_heatmap(self, env=None, source_point=None, xlim=None, ylim=None, grid_size=500, cmap="viridis", wall_alpha=0.35, epsilon=0.1):
        """Heat-map of model-predicted distances with MuJoCo walls overlaid."""
        source_point = self.start_point if source_point is None else source_point
        xlim = self.xlim if xlim is None else xlim
        ylim = self.ylim if ylim is None else ylim
        x_src, y_src = source_point

        # ── 1. build query grid ────────────────────────────────────────────────
        xs = np.linspace(*xlim, grid_size)
        ys = np.linspace(*ylim, grid_size)
        xx, yy = np.meshgrid(xs, ys)
        targets = np.stack([xx.ravel(), yy.ravel()], 1)
        inp = np.hstack([np.full((targets.shape[0], 2), [x_src, y_src]), targets])

        # ── 2. model prediction ───────────────────────────────────────────────
        with torch.no_grad():
            pred = self(torch.as_tensor(inp, dtype=torch.float32)).squeeze().cpu().numpy()
        heatmap = pred.reshape(grid_size, grid_size)

        # ── 3. plot heat-map ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(heatmap, origin="lower", extent=(*xlim, *ylim),
                    cmap=cmap, aspect="equal")
        fig.colorbar(im, ax=ax, label="Predicted distance")

        # ── 4. overlay walls ──────────────────────────────────────────────────
        if env:
            mjm = env.unwrapped.model

            def geom_name(i):
                if hasattr(mjm, "geom_names"):
                    raw = mjm.geom_names[i]
                    return raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
                return mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, i) or ""

            for gid in range(mjm.ngeom):
                name = geom_name(gid).lower()
                if "block" in name:
                    cx, cy = mjm.geom_pos[gid, :2]
                    hx, hy = mjm.geom_size[gid, :2]
                    ax.add_patch(patches.Rectangle(
                        (cx - hx, cy - hy), 2 * hx, 2 * hy,
                        facecolor="black", alpha=wall_alpha, edgecolor="none"))

        # ── 5. decorations ────────────────────────────────────────────────────
        ax.scatter(*source_point, color="red", s=60, edgecolors="black", label="Source")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Distance heat-map with maze")
        ax.grid()
        ax.legend()

        # ── 6. axis limits and equal scaling ──────────────────────────────────
        xeps = epsilon * (xlim[1] - xlim[0])
        yeps = epsilon * (ylim[1] - ylim[0])
        ax.set_xlim(xlim[0] - xeps, xlim[1] + xeps)
        ax.set_ylim(ylim[0] - yeps, ylim[1] + yeps)
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.show()


class SupervisedDistanceEstimator(BaseDistanceEstimator):
    """
    Distance estimator trained with supervised labels (route distances from states to goal).
    """
    def evaluate_from_data(
        self,
        dataset,
        batch_size: int = 8192,
        save_model: bool = False,
        save_path: str = "models/distance_model_sup.pth",
        verbose: bool = True,
    ):
        states_t  = torch.tensor(dataset["eval_states"],    dtype=torch.float32)
        targets_t = torch.tensor(dataset["eval_distances"], dtype=torch.float32).unsqueeze(1)

        n_samples  = states_t.shape[0]
        loss_sum   = 0.0

        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_states  = states_t[start:end]
                batch_targets = targets_t[start:end]

                preds   = self.forward(batch_states)
                loss    = self.loss_fn(preds, batch_targets)
                loss_sum += loss.item() * (end - start)

        eval_loss = loss_sum / n_samples

        if save_model:
            torch.save(self.state_dict(), save_path)
        if verbose:
            print(f"Eval loss: {eval_loss:.4f}")
        return eval_loss

    def train_from_data(
        self,
        dataset,
        epochs: int = 50,
        batch_size: int = 4048,
        eval_every: int = 5,
    ):
        self.start_point, self.xlim, self.ylim = self._get_borders(dataset)
        states_t  = torch.tensor(dataset["train_states"],  dtype=torch.float32)
        targets_t = torch.tensor(dataset["train_distances"], dtype=torch.float32).unsqueeze(1)
        dataset_size = states_t.shape[0]
        eval_losses = []
        avg_train_loss = 0
        for epoch in trange(epochs, desc=f"Training, t_loss={avg_train_loss:.2f}", ncols=100):
            idx = torch.randperm(dataset_size)
            total_loss = 0
            n_batches = 0
            for start in range(0, dataset_size, batch_size):
                end   = start + batch_size
                b_idx = idx[start:end]
                preds = self.forward(states_t[b_idx])
                loss  = self.loss_fn(preds, targets_t[b_idx])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_train_loss = total_loss / n_batches
            wandb.log({"sypervised_losses/train_loss": avg_train_loss}, step=epoch)

            # ── periodic evaluation ───────────────────────────────────────────────
            if epoch % eval_every == 0 or epoch == epochs - 1:
                eval_loss = self.evaluate_from_data(dataset, verbose=False)
                eval_losses.append(eval_loss)
                wandb.log({"sypervised_losses/eval_loss": eval_loss}, step=epoch)

        final_loss = self.loss_fn(self.forward(states_t), targets_t).item()
        print(f"{eval_losses[0]=} -> {eval_losses[-1]=}")
        return final_loss

    @staticmethod
    def _get_borders(dataset):
        """
        Extracts the bounding box and first source point from the dataset.
        
        Args:
            sup_states: numpy array of shape (N, 4) → [x_src, y_src, x_tgt, y_tgt]
        
        Returns:
            start_point: tuple (x, y)
            xlim: (min_x, max_x)
            ylim: (min_y, max_y)
        """
        x_all = np.concatenate([dataset["train_states"][:,0], dataset["train_states"][:,2]])  # x_src and x_tgt
        y_all = np.concatenate([dataset["train_states"][:,1], dataset["train_states"][:,3]])  # y_src and y_tgt

        xlim = (x_all.min(), x_all.max())
        ylim = (y_all.min(), y_all.max())
        start_point = tuple(dataset["train_states"][0][:2])  # (x_src, y_src) of first record

        return start_point, xlim, ylim
    