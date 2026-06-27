from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from scipy.spatial import distance


HEURISTIC_SEEDS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]


@dataclass
class AlgorithmRunResult:
    algorithm: str
    dataset: str
    k_target: int
    seed: int
    order: str
    open_centers: int
    cost: float
    labels: np.ndarray
    X_ordered: np.ndarray
    y_true: np.ndarray
    evaluation_mask: np.ndarray
    cluster_sizes: list[int]
    internal_k: int | None = None
    m_value: int | None = None


def notebook_shuffle(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.arange(len(X))
    rng = np.random.RandomState(seed)
    for _ in range(3):
        order = order[rng.permutation(len(order))]
    return X[order].copy(), y[order].copy()


def euclidean_distance_sq(x: np.ndarray, y: np.ndarray) -> float:
    total = 0.0
    for idx in range(len(x)):
        total += float((x[idx] - y[idx]) ** 2)
    return total


def pairwise_distances_sq(rows: np.ndarray, centers: np.ndarray) -> np.ndarray:
    rr = len(rows)
    cc = len(centers)
    dist = np.zeros((rr, cc), dtype=float)
    for row_idx in range(rr):
        for center_idx in range(cc):
            dist[row_idx, center_idx] = euclidean_distance_sq(rows[row_idx], centers[center_idx])
    return dist


def sorted_center_candidates(centers: list[tuple[int, np.ndarray]], row: np.ndarray) -> list[tuple[float, int]]:
    dist_of_centers: list[tuple[float, int]] = []
    for center_id, center_vec in centers:
        dist_of_centers.append((euclidean_distance_sq(row, center_vec), int(center_id)))
    dist_of_centers.sort()
    return dist_of_centers


def clustering_cost(
    centers: list[tuple[int, np.ndarray]],
    assignments: list[int],
    df_values: np.ndarray,
) -> float:
    center_map = {int(center_id): center_vec for center_id, center_vec in centers}
    cost = 0.0
    idx = 0
    for each in assignments:
        if each != -1:
            cost += euclidean_distance_sq(df_values[idx], center_map[int(each)])
        idx += 1
    return cost


def wstar_helper(centers: list[tuple[int, np.ndarray]]) -> float:
    distances_values = []
    for idx_i in range(len(centers)):
        for idx_j in range(len(centers)):
            if idx_i != idx_j:
                distances_values.append(
                    euclidean_distance_sq(centers[idx_i][1], centers[idx_j][1]),
                )
    return float(np.min(np.array(distances_values, dtype=float)))


def wstar_helper_heur(centers: list[tuple[int, np.ndarray]]) -> float:
    nearest_distances = []
    for idx_i in range(len(centers)):
        min_dist = float("inf")
        for idx_j in range(len(centers)):
            if idx_i != idx_j:
                dist_value = euclidean_distance_sq(centers[idx_i][1], centers[idx_j][1])
                if dist_value < min_dist:
                    min_dist = dist_value
        nearest_distances.append(min_dist)
    nearest_distances = np.sort(np.array(nearest_distances, dtype=float))[:10]
    return float(np.mean(nearest_distances))


def cap_heur(
    k_target: int,
    x_values: np.ndarray,
    capacities: list[list[float]],
    mode: str,
) -> float:
    n_row, _ = x_values.shape
    rand_indices = np.random.choice(n_row, size=k_target)
    centroids = x_values[rand_indices]

    for _ in range(100):
        distances_to_centroids = pairwise_distances_sq(x_values, centroids)
        cluster_assignment = []
        capacities_iter = np.array(capacities, dtype=np.int64)

        for idx in range(len(x_values)):
            sorted_center_ids = np.argsort(distances_to_centroids[idx])
            for center_id in sorted_center_ids:
                if capacities_iter[int(center_id)][1] > 0:
                    cluster_assignment.append(center_id)
                    capacities_iter[int(center_id)][1] -= 1
                    break

        cluster_assignment = np.array(cluster_assignment)
        new_centroids = []
        for center_idx in range(k_target):
            assign = np.array([x_values[cluster_assignment == center_idx]])
            med = np.mean(assign, axis=1)
            new_centroids.append(med)

        new_centroids = np.array(new_centroids)
        new_centroids = new_centroids.reshape(k_target, len(x_values[0]))
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    cost_values = []
    for center_idx in range(k_target):
        cluster_data = x_values[cluster_assignment == center_idx]
        distances_cluster = distance.cdist(cluster_data, [centroids[center_idx]], "euclidean")
        cost_values.append(np.sum(distances_cluster**2))

    if mode == "max":
        return float(np.max(cost_values))
    return float(np.sum(cost_values))


def lower_bound_cap_heuristic(
    k_target: int,
    df_values: np.ndarray,
    mode: str,
    capacity_multiplier: float = 1.0,
    heuristic_seeds: list[int] | None = None,
) -> list[float]:
    heuristic_seeds = HEURISTIC_SEEDS if heuristic_seeds is None else heuristic_seeds
    w_list = []
    for seed in heuristic_seeds:
        random.seed(seed)
        np.random.seed(seed)
        capacities = []
        for idx in range(k_target):
            capacities.append([idx, np.ceil(capacity_multiplier * len(df_values) / k_target)])
        w_list.append(cap_heur(k_target, df_values, capacities, mode=mode))
    return w_list


def n_cr(n_value: int, r_value: int) -> int:
    if r_value > n_value:
        return 0
    if r_value == 0 or n_value == r_value:
        return 1

    res = 0.0
    for idx in range(r_value):
        res += math.log(n_value - idx) - math.log(idx + 1)
    return round(math.exp(res))


def coupon_collector_non_uniform(
    capacities: list[tuple[int, float]],
    k_target: int,
) -> int:
    prob = []
    total = 0.0
    for each in capacities:
        total += each[1]
        prob.append(each[1])

    prob = np.array(prob, dtype=float)
    prob = np.round(prob / total, 2)

    expec = 0.0
    for m_value in range(1, k_target + 1):
        multiplier = (-1) ** (m_value - 1)
        sum_inner = (1.0 * n_cr(k_target, m_value)) / (prob[0] * m_value)
        expec += multiplier * sum_inner
    return int(np.ceil(np.round(expec, 2)))


def canonicalize_labels(labels: np.ndarray) -> np.ndarray:
    mapping: dict[int, int] = {}
    canonical = []
    next_id = 0
    for label in labels.tolist():
        label = int(label)
        if label not in mapping:
            mapping[label] = next_id
            next_id += 1
        canonical.append(mapping[label])
    return np.array(canonical, dtype=int)


def cluster_sizes_from_labels(labels: np.ndarray) -> list[int]:
    if len(labels) == 0:
        return []
    _, counts = np.unique(labels, return_counts=True)
    return sorted(int(each) for each in counts.tolist())


def stabilize_positive_cost(value: float, eps: float = 1e-12) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        return eps
    return value
