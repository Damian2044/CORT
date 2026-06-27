from __future__ import annotations

from pathlib import Path

import numpy as np

from algoritmos.ejecucion import ejecutar_algoritmo
from algoritmos.utilidades import DatasetBundle
from metricas import construir_fila_error, construir_fila_resultado
from recursos import medir_tiempo_y_ram, ocultar_advertencias


def ejecutar_tarea_npz(tarea: dict[str, object]) -> dict[str, object]:
    """Worker para modalidades paralelas: carga una copia temporal y ejecuta una tarea."""
    ocultar_advertencias()

    dataset = str(tarea["dataset"])
    algoritmo = str(tarea["algoritmo"])
    semilla = int(tarea["semilla"])
    modalidad = str(tarea["modalidad"])
    workers = int(tarea["workers"])
    try:
        with np.load(Path(str(tarea["npz"])), allow_pickle=True) as data:
            x_values = np.asarray(data["X"], dtype=float)
            y_values = np.asarray(data["y"], dtype=int)
            cardinalidad = tuple(int(v) for v in np.asarray(data["cardinalidad"], dtype=int).tolist())
            feature_names = tuple(str(v) for v in data["feature_names"].tolist())
            class_names = tuple(str(v) for v in data["class_names"].tolist())

        bundle = DatasetBundle(dataset, x_values, y_values, feature_names, class_names)

        resultado, tiempo, ram_max, inc_ram = medir_tiempo_y_ram(
            lambda: ejecutar_algoritmo(algoritmo, bundle, cardinalidad, semilla),
            intervalo_segundos=float(tarea["intervalo_ram"]),
        )
        return construir_fila_resultado(
            dataset=bundle,
            resultado=resultado,
            cardinalidad_objetivo=cardinalidad,
            semilla=semilla,
            limite_silueta=int(tarea["limite_silueta"]),
            tiempo_algoritmo_s=tiempo,
            ram_max_mb=ram_max,
            incremento_ram_mb=inc_ram,
            modalidad=modalidad,
            workers=workers,
        )
    except Exception as exc:
        return construir_fila_error(dataset, algoritmo, semilla, exc, modalidad, workers)
