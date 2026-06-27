from __future__ import annotations

import random

import numpy as np

from .utilidades import (
    AlgorithmRunResult,
    DatasetBundle,
    cluster_sizes_from_labels,
    clustering_cost,
    coupon_collector_non_uniform,
    lower_bound_cap_heuristic,
    notebook_shuffle,
    sorted_center_candidates,
    stabilize_positive_cost,
    wstar_helper,
    wstar_helper_heur,
)


ALGORITMOS_SOTA = ("LIB", "LIBH", "COCA", "COCH")


def ejecutar_algoritmo_sota(
    algoritmo: str,
    dataset: DatasetBundle,
    k_objetivo: int,
    semilla: int,
) -> AlgorithmRunResult:
    if algoritmo not in ALGORITMOS_SOTA:
        raise ValueError(f"Algoritmo SOTA no soportado: {algoritmo}")

    return run_sota_algorithm(
        algorithm=algoritmo,
        dataset=dataset,
        k_target=k_objetivo,
        seed=semilla,
        order="arbitrary",
    )


def run_sota_algorithm(
    algorithm: str,
    dataset: DatasetBundle,
    k_target: int,
    seed: int,
    order: str = "random",
) -> AlgorithmRunResult:
    if algorithm == "LIB":
        return _run_lib_base(dataset, k_target, seed, order)
    if algorithm == "LIBH":
        return _run_libh_base(dataset, k_target, seed, order)
    if algorithm == "COCA":
        return _run_coca_base(dataset, k_target, seed, order)
    if algorithm == "COCH":
        return _run_coch_base(dataset, k_target, seed, order)
    raise ValueError(f"Algoritmo no soportado: {algorithm}")


def _prepare_data(dataset: DatasetBundle, seed: int, order: str) -> tuple[np.ndarray, np.ndarray]:
    if order == "random":
        return notebook_shuffle(dataset.X, dataset.y, seed)
    return dataset.X.copy(), dataset.y.copy()


def _run_lib_base(
    dataset: DatasetBundle,
    k_target: int,
    seed: int,
    order: str,
) -> AlgorithmRunResult:
    x_values, y_true = _prepare_data(dataset, seed, order)
    random_state = random.Random(seed)

    n_samples, n_features = x_values.shape
    center_ids = np.empty(n_samples, dtype=np.int64)
    center_values = np.empty((n_samples, n_features), dtype=float)
    center_count = 0
    centers_opened_temp = 0
    centers_opened = 0
    assignments: list[int] = []
    labels = np.full(n_samples, -1, dtype=int)
    evaluation_mask = np.ones(n_samples, dtype=bool)

    for idx in range(0, k_target + 1):
        center_ids[center_count] = idx
        center_values[center_count] = x_values[idx]
        center_count += 1
        assignments.append(-1)
        labels[idx] = centers_opened
        centers_opened += 1
        centers_opened_temp += 1

    center_cost = stabilize_positive_cost(
        wstar_helper(_as_center_list(center_ids, center_values, center_count)) / k_target,
    )

    for idx in range(k_target + 1, len(x_values)):
        if centers_opened_temp >= 3 * k_target * (1 + np.log(idx)):
            center_cost = center_cost * 2
            centers_opened_temp = 0

        current = x_values[idx]
        deltas = np.sum((center_values[:center_count] - current) ** 2, axis=1)
        best_pos = int(np.argmin(deltas))
        closest_center_index = int(center_ids[best_pos])
        closest_center_dist = float(deltas[best_pos])

        prob_center = np.round(closest_center_dist / stabilize_positive_cost(center_cost), 3)
        curr_prob = np.round(random_state.random(), 3)

        if curr_prob < prob_center:
            center_ids[center_count] = centers_opened
            center_values[center_count] = current
            center_count += 1
            assignments.append(-1)
            labels[idx] = centers_opened
            centers_opened += 1
            centers_opened_temp += 1
        else:
            assignments.append(closest_center_index)
            labels[idx] = closest_center_index

    centers = _as_center_list(center_ids, center_values, center_count)
    cost = np.round(clustering_cost(centers, assignments, x_values), 3)
    return AlgorithmRunResult(
        algorithm="LIB",
        dataset=dataset.name,
        k_target=k_target,
        seed=seed,
        order=order,
        open_centers=centers_opened,
        cost=float(cost),
        labels=labels,
        X_ordered=x_values,
        y_true=y_true,
        evaluation_mask=evaluation_mask,
        cluster_sizes=cluster_sizes_from_labels(labels[evaluation_mask]),
        internal_k=k_target,
    )


def _run_libh_base(
    dataset: DatasetBundle,
    k_target: int,
    seed: int,
    order: str,
) -> AlgorithmRunResult:
    internal_k = int(np.ceil((k_target - 15) / 5)) if k_target > 15 else k_target
    x_values, y_true = _prepare_data(dataset, seed, order)
    random_state = random.Random(seed)

    n_samples, n_features = x_values.shape
    center_ids = np.empty(n_samples, dtype=np.int64)
    center_values = np.empty((n_samples, n_features), dtype=float)
    center_count = 0
    centers_opened_temp = 0
    centers_opened = 0
    assignments: list[int] = []
    labels = np.full(n_samples, -1, dtype=int)
    evaluation_mask = np.ones(n_samples, dtype=bool)

    for idx in range(0, internal_k + 1):
        center_ids[center_count] = idx
        center_values[center_count] = x_values[idx]
        center_count += 1
        assignments.append(-1)
        labels[idx] = centers_opened
        centers_opened += 1
        centers_opened_temp += 1

    center_cost = stabilize_positive_cost(
        wstar_helper_heur(_as_center_list(center_ids, center_values, center_count)),
    )

    for idx in range(internal_k + 1, len(x_values)):
        if centers_opened_temp >= internal_k:
            center_cost = center_cost * 10
            centers_opened_temp = 0

        current = x_values[idx]
        deltas = np.sum((center_values[:center_count] - current) ** 2, axis=1)
        best_pos = int(np.argmin(deltas))
        closest_center_index = int(center_ids[best_pos])
        closest_center_dist = float(deltas[best_pos])

        prob_center = np.round(closest_center_dist / stabilize_positive_cost(center_cost), 3)
        curr_prob = np.round(random_state.random(), 3)

        if curr_prob < prob_center:
            center_ids[center_count] = centers_opened
            center_values[center_count] = current
            center_count += 1
            assignments.append(-1)
            labels[idx] = centers_opened
            centers_opened += 1
            centers_opened_temp += 1
        else:
            assignments.append(closest_center_index)
            labels[idx] = closest_center_index

    centers = _as_center_list(center_ids, center_values, center_count)
    cost = np.round(clustering_cost(centers, assignments, x_values), 3)
    return AlgorithmRunResult(
        algorithm="LIBH",
        dataset=dataset.name,
        k_target=k_target,
        seed=seed,
        order=order,
        open_centers=centers_opened,
        cost=float(cost),
        labels=labels,
        X_ordered=x_values,
        y_true=y_true,
        evaluation_mask=evaluation_mask,
        cluster_sizes=cluster_sizes_from_labels(labels[evaluation_mask]),
        internal_k=internal_k,
    )


def _run_coca_base(
    dataset: DatasetBundle,
    k_target: int,
    seed: int,
    order: str,
) -> AlgorithmRunResult:
    x_values, y_true = _prepare_data(dataset, seed, order)

    capacities_optimal = []
    for idx in range(k_target):
        capacities_optimal.append((idx, np.ceil(len(x_values) / k_target)))

    m_value = min(coupon_collector_non_uniform(capacities_optimal, k_target), len(x_values))
    nestd = m_value
    heuristic_cost = lower_bound_cap_heuristic(
        k_target,
        x_values[: m_value + 1],
        mode="sum",
        heuristic_seeds=[seed],
    )[0]
    random_state = random.Random(seed)

    centers: list[tuple[int, np.ndarray]] = []
    capacities: list[float] = []
    labels = np.full(len(x_values), -1, dtype=int)
    start_index = m_value
    evaluation_mask = np.ones(len(x_values), dtype=bool)

    centers_opened = 0
    centers_opened_temp = 0
    assignments: list[int] = []

    for idx in range(0, m_value):
        centers.append((idx, x_values[idx].copy()))
        capacities.append(np.ceil(len(x_values) / k_target))
        assignments.append(-1)
        labels[idx] = idx
        centers_opened += 1
        centers_opened_temp += 1

    center_cost = stabilize_positive_cost(heuristic_cost / (k_target * np.log(nestd)))

    for idx in range(start_index, len(x_values)):
        if idx > nestd:
            nestd = nestd * 2

        if centers_opened_temp >= np.ceil(3 * k_target * (1 + np.log(nestd))):
            center_cost = center_cost * 2
            centers_opened_temp = 0

        current = x_values[idx]
        dist_of_centers = sorted_center_candidates(centers, current)
        flag_assign = False

        for dist_value, closest_center_index in dist_of_centers:
            prob_center = np.round(dist_value / stabilize_positive_cost(center_cost), 3)
            curr_prob = np.round(random_state.random(), 3)
            if curr_prob < prob_center:
                centers.append((centers_opened, current.copy()))
                capacities.append(np.ceil(len(x_values) / k_target))
                assignments.append(-1)
                labels[idx] = centers_opened
                centers_opened += 1
                centers_opened_temp += 1
                flag_assign = True
                break

            if capacities[closest_center_index] > 0:
                capacities[closest_center_index] -= 1
                assignments.append(closest_center_index)
                labels[idx] = closest_center_index
                flag_assign = True
                break

        if flag_assign is False:
            centers.append((centers_opened, current.copy()))
            capacities.append(np.ceil(len(x_values) / k_target))
            assignments.append(-1)
            labels[idx] = centers_opened
            centers_opened += 1
            centers_opened_temp += 1

    cost = np.round(clustering_cost(centers, assignments, x_values), 3)
    return AlgorithmRunResult(
        algorithm="COCA",
        dataset=dataset.name,
        k_target=k_target,
        seed=seed,
        order=order,
        open_centers=centers_opened,
        cost=float(cost),
        labels=labels,
        X_ordered=x_values,
        y_true=y_true,
        evaluation_mask=evaluation_mask,
        cluster_sizes=cluster_sizes_from_labels(labels[evaluation_mask]),
        m_value=m_value,
    )


def _run_coch_base(
    dataset: DatasetBundle,
    k_target: int,
    seed: int,
    order: str,
) -> AlgorithmRunResult:
    x_values, y_true = _prepare_data(dataset, seed, order)

    m_value = k_target + 1
    heuristic_cost = lower_bound_cap_heuristic(
        k_target,
        x_values[: m_value + 1],
        mode="sum",
        heuristic_seeds=[seed],
    )[0]
    random_state = random.Random(seed)

    centers: list[tuple[int, np.ndarray]] = []
    capacities: list[float] = []
    labels = np.full(len(x_values), -1, dtype=int)
    evaluation_mask = np.ones(len(x_values), dtype=bool)

    centers_opened = 0
    centers_opened_temp = 0
    assignments: list[int] = []

    for idx in range(0, m_value):
        centers.append((idx, x_values[idx].copy()))
        capacities.append(np.ceil(len(x_values) / k_target))
        assignments.append(-1)
        labels[idx] = idx
        centers_opened += 1
        centers_opened_temp += 1

    center_cost = stabilize_positive_cost(heuristic_cost / k_target)

    for idx in range(m_value, len(x_values)):
        if centers_opened_temp >= k_target:
            center_cost = center_cost * 10
            centers_opened_temp = 0

        current = x_values[idx]
        dist_of_centers = sorted_center_candidates(centers, current)
        flag_assign = False

        for dist_value, closest_center_index in dist_of_centers:
            prob_center = np.round(dist_value / stabilize_positive_cost(center_cost), 3)
            curr_prob = np.round(random_state.random(), 3)
            if curr_prob < prob_center:
                centers.append((centers_opened, current.copy()))
                capacities.append(np.ceil(len(x_values) / k_target))
                assignments.append(-1)
                labels[idx] = centers_opened
                centers_opened += 1
                centers_opened_temp += 1
                flag_assign = True
                break

            if capacities[closest_center_index] > 0:
                capacities[closest_center_index] -= 1
                assignments.append(closest_center_index)
                labels[idx] = closest_center_index
                flag_assign = True
                break

        if flag_assign is False:
            centers.append((centers_opened, current.copy()))
            capacities.append(np.ceil(len(x_values) / k_target))
            assignments.append(-1)
            labels[idx] = centers_opened
            centers_opened += 1
            centers_opened_temp += 1

    cost = np.round(clustering_cost(centers, assignments, x_values), 3)
    return AlgorithmRunResult(
        algorithm="COCH",
        dataset=dataset.name,
        k_target=k_target,
        seed=seed,
        order=order,
        open_centers=centers_opened,
        cost=float(cost),
        labels=labels,
        X_ordered=x_values,
        y_true=y_true,
        evaluation_mask=evaluation_mask,
        cluster_sizes=cluster_sizes_from_labels(labels[evaluation_mask]),
        m_value=m_value,
    )


def _as_center_list(
    center_ids: np.ndarray,
    center_values: np.ndarray,
    center_count: int,
) -> list[tuple[int, np.ndarray]]:
    return [
        (int(center_ids[pos]), center_values[pos].copy())
        for pos in range(center_count)
    ]

