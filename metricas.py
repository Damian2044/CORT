from __future__ import annotations

import ast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from algoritmos.utilidades import AlgorithmRunResult, DatasetBundle


def cardinalidad_objetivo_texto(y_values: np.ndarray, k: int) -> str:
    valores = np.bincount(np.asarray(y_values, dtype=int), minlength=int(k)).astype(int)
    return ",".join(str(int(v)) for v in valores.tolist())


def calcular_silueta(x_values: np.ndarray, labels: np.ndarray, semilla: int, limite: int | None) -> float:
    etiquetas = np.unique(labels)
    if len(etiquetas) < 2 or len(etiquetas) >= len(labels):
        return float("nan")
    sample_size = int(limite) if limite is not None and len(labels) > int(limite) else None
    return float(silhouette_score(x_values, labels, sample_size=sample_size, random_state=semilla))


def construir_fila_resultado(
    dataset: DatasetBundle,
    resultado: AlgorithmRunResult,
    cardinalidad_objetivo: tuple[int, ...],
    semilla: int,
    limite_silueta: int | None,
    tiempo_algoritmo_s: float,
    ram_max_mb: float,
    incremento_ram_mb: float,
    modalidad: str,
    workers: int,
) -> dict[str, object]:
    mascara = resultado.evaluation_mask
    x_eval = resultado.X_ordered[mascara]
    y_eval = resultado.y_true[mascara]
    labels_eval = resultado.labels[mascara]
    return {
        "dataset": dataset.name,
        "algoritmo": resultado.algorithm,
        "semilla": int(semilla),
        "estado": "ok",
        "error": "",
        "modalidad": modalidad,
        "workers": int(workers),
        "instancias": int(len(resultado.X_ordered)),
        "n_eval": int(mascara.sum()),
        "features": int(resultado.X_ordered.shape[1]),
        "n_clusters": int(len(cardinalidad_objetivo)),
        "clases": int(len(np.unique(resultado.y_true))),
        "cardinalidad_objetivo": ",".join(str(int(v)) for v in cardinalidad_objetivo),
        "silhouette": calcular_silueta(x_eval, labels_eval, semilla, limite_silueta),
        "ari": float(adjusted_rand_score(y_eval, labels_eval)),
        "ami": float(adjusted_mutual_info_score(y_eval, labels_eval)),
        "nmi": float(normalized_mutual_info_score(y_eval, labels_eval)),
        "open_centers": int(resultado.open_centers),
        "n_clusters_eval": int(len(np.unique(labels_eval))),
        "cost": float(resultado.cost),
        "tiempo_algoritmo_s": float(tiempo_algoritmo_s),
        "ram_max_mb": float(ram_max_mb),
        "incremento_ram_mb": float(incremento_ram_mb),
        "cluster_sizes": str(resultado.cluster_sizes),
        "internal_k": resultado.internal_k,
        "m_value": resultado.m_value,
    }


def construir_fila_error(dataset: str, algoritmo: str, semilla: int, error: Exception, modalidad: str, workers: int) -> dict[str, object]:
    return {
        "dataset": dataset,
        "algoritmo": algoritmo,
        "semilla": int(semilla),
        "estado": "error",
        "error": str(error),
        "modalidad": modalidad,
        "workers": int(workers),
    }


def resumen_cardinalidades(series: pd.Series) -> str:
    valores: list[int] = []
    for item in series.tolist():
        try:
            parsed = ast.literal_eval(str(item))
        except Exception:
            parsed = []
        if isinstance(parsed, (list, tuple, np.ndarray)):
            valores.extend(int(v) for v in parsed)
    if not valores:
        return "-"
    q = np.quantile(np.asarray(valores, dtype=float), [0, 0.25, 0.5, 0.75, 1], method="nearest")
    return "|".join(str(int(v)) for v in q.tolist())
