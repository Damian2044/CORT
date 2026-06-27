from __future__ import annotations

import numpy as np

from .cort import CORT
from .sota import ejecutar_algoritmo_sota
from .utilidades import AlgorithmRunResult, DatasetBundle, cluster_sizes_from_labels


def ejecutar_algoritmo(
    algoritmo: str,
    dataset: DatasetBundle,
    cardinalidad_objetivo: tuple[int, ...],
    semilla: int,
) -> AlgorithmRunResult:
    """Ejecuta un algoritmo con un dataset ya barajado."""
    if algoritmo in {"LIB", "LIBH", "COCA", "COCH"}:
        return ejecutar_algoritmo_sota(
            algoritmo=algoritmo,
            dataset=dataset,
            k_objetivo=len(cardinalidad_objetivo),
            semilla=semilla,
        )
    if algoritmo == "CORT":
        return _ejecutar_cort(dataset, cardinalidad_objetivo, semilla)
    raise ValueError(f"Algoritmo no soportado: {algoritmo}")


def _ejecutar_cort(
    dataset: DatasetBundle,
    cardinalidad_objetivo: tuple[int, ...],
    semilla: int,
) -> AlgorithmRunResult:
    modelo = CORT(k=len(cardinalidad_objetivo), cardinalidades=list(cardinalidad_objetivo))
    labels = []
    for punto in dataset.X:
        labels.append(modelo.procesarPunto(punto))

    labels_arr = np.asarray(labels, dtype=int)
    mascara = labels_arr >= 0
    return AlgorithmRunResult(
        algorithm="CORT",
        dataset=dataset.name,
        k_target=len(cardinalidad_objetivo),
        seed=semilla,
        order="arbitrary",
        open_centers=int(modelo.KFundados),
        cost=float("nan"),
        labels=labels_arr,
        X_ordered=dataset.X,
        y_true=dataset.y,
        evaluation_mask=mascara,
        cluster_sizes=cluster_sizes_from_labels(labels_arr[mascara]),
        internal_k=len(cardinalidad_objetivo),
    )
