import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

def get_borders_by_trajectory(trajectories):
    all_points = np.concatenate([np.array(traj) for traj in trajectories], axis=0)
    x_all, y_all = all_points[:, 0], all_points[:, 1]
    
    xlim = (x_all.min(), x_all.max())
    ylim = (y_all.min(), y_all.max())
    start_point = tuple(trajectories[0][0])

    return start_point, xlim, ylim

def trajectories_to_dataset(
    trajectories: List[List[Tuple[float, float]]],
    samples: Optional[int] = None,
    split_ratio: float = 0.8,
    random_state: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Build a (train, eval) supervised dataset from 2-D trajectories.

    Returns
    -------
    dict containing:
        - train_states      (N_train, 4)  [x_src, y_src, x_tgt, y_tgt]
        - train_distances   (N_train,)
        - eval_states       (N_eval, 4)  [x_src, y_src, x_tgt, y_tgt]
        - eval_distances    (N_eval,)
    """
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    src_list, tgt_list, dist_list = [], [], []

    for traj in tqdm(trajectories, desc="Building dataset", ncols=100):
        coords = np.asarray(traj, dtype=float)
        T = len(coords)
        if T < 2:
            continue

        step_dists = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        travel_dists = np.concatenate(([0.0], np.cumsum(step_dists)))

        i_idx, j_idx = np.triu_indices(T)
        src_list.append(coords[i_idx])
        tgt_list.append(coords[j_idx])
        dist_list.append(travel_dists[j_idx] - travel_dists[i_idx])

    if not src_list:  # no usable trajectories
        raise ValueError("No trajectories with length ≥ 2 were provided.")

    sources = np.concatenate(src_list)          # (N, 2)
    targets = np.concatenate(tgt_list)          # (N, 2)
    distances = np.concatenate(dist_list)       # (N,)
    states = np.hstack([sources, targets])      # (N, 4)

    # Shuffle & optional down-sampling
    perm = rng.permutation(len(states))
    if samples is not None and samples > 0:
        perm = perm[: samples]
    states, distances = states[perm], distances[perm]

    # Train / eval split
    split = int(len(states) * split_ratio)
    train_states, eval_states = states[:split], states[split:]
    train_distances, eval_distances = distances[:split], distances[split:]

    print(f"Trajectories processed : {len(trajectories)}")
    print(f"Generated samples      : {len(states)} "
          f"(train {len(train_states)} / eval {len(eval_states)})")

    return {
        "train_states": train_states,
        "train_distances": train_distances,
        "eval_states": eval_states,
        "eval_distances": eval_distances,
    }
