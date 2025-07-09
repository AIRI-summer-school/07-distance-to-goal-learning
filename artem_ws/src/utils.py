import numpy as np
from tqdm import tqdm


def trajectories_to_dataset(trajectories: list[list[tuple[float, float]]]) -> dict:
    """
    Converts a list of trajectories into a single dataset.
    Each trajectory contributes approx T²/2 samples where T is its length.
    """
    source_x, source_y, target_x, target_y, distances = [], [], [], [], []

    for traj in tqdm(trajectories, desc="Building dataset", ncols=100):
        translations = np.array(traj)
        T = len(translations)
        if T < 2:
            continue  # skip too-short trajectories

        diffs = np.linalg.norm(np.diff(translations, axis=0), axis=1)
        travel_dists = np.concatenate([[0], np.cumsum(diffs)])

        for i in range(T):
            for j in range(i, T):
                sx, sy = translations[i]
                tx, ty = translations[j]
                source_x.append(sx)
                source_y.append(sy)
                target_x.append(tx)
                target_y.append(ty)
                distances.append(travel_dists[j] - travel_dists[i])

    return {
        "source_x": np.array(source_x),
        "source_y": np.array(source_y),
        "target_x": np.array(target_x),
        "target_y": np.array(target_y),
        "travel_distance": np.array(distances),
    }