from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from algoritmos.ejecucion import ejecutar_algoritmo
from consola import filtrar_warnings_esperados
from cargador_datasets import cargar_dataset, mezclar_dataset
from ejecutor_tareas import ejecutar_tarea_npz
from metricas import construir_fila_error, construir_fila_resultado
from recursos import medir_tiempo_y_ram
from reportes import (
    construir_resumen,
    construir_resumen_compacto,
    crear_carpeta_salida,
    guardar_salidas,
    ordenar_resultados,
)


RAIZ_PROYECTO = Path(__file__).resolve().parent
RAIZ_RESULTADOS = RAIZ_PROYECTO / "resultados"


def contexto_multiproceso():
    """Usa spawn en Windows y fork en Linux cuando esta disponible."""
    if sys.platform == "win32":
        return get_context("spawn")
    try:
        return get_context("fork")
    except ValueError:
        return get_context("spawn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runner principal de CORT experimentos.")
    parser.add_argument("--modalidad", choices=["secuencial", "paralelo", "paralelo_aislado", "paralelo_rapido"])
    parser.add_argument("--workers", type=int)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--algoritmos", nargs="*")
    parser.add_argument("--semillas", type=int)
    parser.add_argument("--prefijo-salida")
    parser.add_argument("--no-guardar", action="store_true")
    return parser.parse_args()


def formato_duracion(segundos: float) -> str:
    total = int(round(segundos))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def construir_configuracion_salida(
    datasets: tuple[str, ...],
    algoritmos: tuple[str, ...],
    semillas: tuple[int, ...],
    modalidad: str,
    workers: int,
    prefijo: str,
    guardar: bool,
) -> dict[str, object]:
    """Arma la configuracion que se guarda junto a los resultados."""
    return {
        "datasets_a_correr": list(datasets),
        "algoritmos_a_correr": list(algoritmos),
        "metricas_a_calcular": list(config.METRICAS_A_CALCULAR),
        "semilla_inicial": int(config.SEMILLA_INICIAL),
        "cantidad_semillas": len(semillas),
        "semillas": list(semillas),
        "semilla_sinteticos": config.SEMILLA_SINTETICOS,
        "limite_silueta": int(config.LIMITE_SILUETA),
        "modalidad_ejecucion": modalidad,
        "workers": int(workers),
        "prefijo_salida": prefijo,
        "guardar_resultados": bool(guardar),
        "intervalo_ram_segundos": float(config.INTERVALO_RAM_SEGUNDOS),
    }


def guardar_npz_temporal(path: Path, dataset_cargado) -> None:
    np.savez_compressed(
        path,
        X=dataset_cargado.bundle.X,
        y=dataset_cargado.bundle.y,
        cardinalidad=np.asarray(dataset_cargado.cardinalidad_objetivo, dtype=int),
        feature_names=np.asarray(dataset_cargado.bundle.feature_names, dtype=object),
        class_names=np.asarray(dataset_cargado.bundle.class_names, dtype=object),
    )


def crear_tareas(
    datasets: tuple[str, ...],
    algoritmos: tuple[str, ...],
    semillas: tuple[int, ...],
    modalidad: str,
    workers: int,
    temp_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tareas: list[dict[str, object]] = []
    errores: list[dict[str, object]] = []
    for dataset_name in datasets:
        try:
            semilla_base = config.SEMILLA_SINTETICOS
            base = cargar_dataset(dataset_name, semilla_sinteticos=semilla_base)
        except Exception as exc:
            for semilla in semillas:
                for algoritmo in algoritmos:
                    errores.append(construir_fila_error(dataset_name, algoritmo, semilla, exc, modalidad, workers))
            continue

        for semilla in semillas:
            try:
                semilla_sint = semilla if config.SEMILLA_SINTETICOS is None else config.SEMILLA_SINTETICOS
                if str(dataset_name).startswith("Synthetic") and semilla_sint != config.SEMILLA_SINTETICOS:
                    base_semilla = cargar_dataset(dataset_name, semilla_sinteticos=semilla_sint)
                else:
                    base_semilla = base
                mezclado = mezclar_dataset(base_semilla, semilla)
                npz_path = temp_dir / f"{dataset_name.replace('/', '_')}_semilla_{semilla}.npz"
                guardar_npz_temporal(npz_path, mezclado)
            except Exception as exc:
                for algoritmo in algoritmos:
                    errores.append(construir_fila_error(dataset_name, algoritmo, semilla, exc, modalidad, workers))
                continue

            for algoritmo in algoritmos:
                tareas.append(
                    {
                        "dataset": dataset_name,
                        "algoritmo": algoritmo,
                        "semilla": int(semilla),
                        "npz": str(npz_path),
                        "modalidad": modalidad,
                        "workers": int(workers),
                        "limite_silueta": int(config.LIMITE_SILUETA),
                        "intervalo_ram": float(config.INTERVALO_RAM_SEGUNDOS),
                    },
                )
    return tareas, errores


def ejecutar_secuencial(
    datasets: tuple[str, ...],
    algoritmos: tuple[str, ...],
    semillas: tuple[int, ...],
    modalidad: str,
    workers: int,
) -> list[dict[str, object]]:
    filas: list[dict[str, object]] = []
    total = len(datasets) * len(algoritmos) * len(semillas)
    with tqdm(total=total, desc="Experimento", unit="ejecucion") as barra:
        for dataset_name in datasets:
            try:
                base = cargar_dataset(dataset_name, semilla_sinteticos=config.SEMILLA_SINTETICOS)
            except Exception as exc:
                for semilla in semillas:
                    for algoritmo in algoritmos:
                        filas.append(construir_fila_error(dataset_name, algoritmo, semilla, exc, modalidad, workers))
                        barra.update(1)
                continue

            for semilla in semillas:
                try:
                    semilla_sint = semilla if config.SEMILLA_SINTETICOS is None and dataset_name.startswith("Synthetic") else config.SEMILLA_SINTETICOS
                    base_semilla = cargar_dataset(dataset_name, semilla_sint) if semilla_sint != config.SEMILLA_SINTETICOS else base
                    mezclado = mezclar_dataset(base_semilla, semilla)
                except Exception as exc:
                    for algoritmo in algoritmos:
                        filas.append(construir_fila_error(dataset_name, algoritmo, semilla, exc, modalidad, workers))
                        barra.update(1)
                    continue

                for algoritmo in algoritmos:
                    barra.set_postfix(dataset=dataset_name, algoritmo=algoritmo, semilla=semilla)
                    try:
                        resultado, tiempo, ram_max, inc_ram = medir_tiempo_y_ram(
                            lambda a=algoritmo, d=mezclado: ejecutar_algoritmo(a, d.bundle, d.cardinalidad_objetivo, semilla),
                            intervalo_segundos=float(config.INTERVALO_RAM_SEGUNDOS),
                        )
                        filas.append(
                            construir_fila_resultado(
                                dataset=mezclado.bundle,
                                resultado=resultado,
                                cardinalidad_objetivo=mezclado.cardinalidad_objetivo,
                                semilla=semilla,
                                limite_silueta=int(config.LIMITE_SILUETA),
                                tiempo_algoritmo_s=tiempo,
                                ram_max_mb=ram_max,
                                incremento_ram_mb=inc_ram,
                                modalidad=modalidad,
                                workers=workers,
                            ),
                        )
                    except Exception as exc:
                        filas.append(construir_fila_error(dataset_name, algoritmo, semilla, exc, modalidad, workers))
                    barra.update(1)
    return filas


def ejecutar_paralelo(
    datasets: tuple[str, ...],
    algoritmos: tuple[str, ...],
    semillas: tuple[int, ...],
    modalidad: str,
    workers: int,
) -> list[dict[str, object]]:
    filas: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="cort_experimentos_") as tmp:
        tareas, errores = crear_tareas(datasets, algoritmos, semillas, modalidad, workers, Path(tmp))
        filas.extend(errores)
        total = len(tareas) + len(errores)
        contexto = contexto_multiproceso()
        pool_kwargs = {"processes": max(1, int(workers))}
        if modalidad == "paralelo_aislado":
            pool_kwargs["maxtasksperchild"] = 1

        with tqdm(total=total, desc="Experimento", unit="ejecucion") as barra:
            barra.update(len(errores))
            with contexto.Pool(**pool_kwargs) as pool:
                for fila in pool.imap_unordered(ejecutar_tarea_npz, tareas):
                    filas.append(fila)
                    barra.set_postfix(
                        dataset=fila.get("dataset", ""),
                        algoritmo=fila.get("algoritmo", ""),
                        semilla=fila.get("semilla", ""),
                    )
                    barra.update(1)
    return filas


def main() -> None:
    filtrar_warnings_esperados()

    args = parse_args()
    datasets = tuple(args.datasets) if args.datasets else tuple(config.DATASETS_A_CORRER)
    algoritmos = tuple(args.algoritmos) if args.algoritmos else tuple(config.ALGORITMOS_A_CORRER)
    cantidad_semillas = int(args.semillas) if args.semillas is not None else int(config.CANTIDAD_SEMILLAS)
    semillas = tuple(range(int(config.SEMILLA_INICIAL), int(config.SEMILLA_INICIAL) + cantidad_semillas))
    modalidad = str(args.modalidad or config.MODALIDAD_EJECUCION)
    if modalidad == "paralelo_rapido":
        modalidad = "paralelo"
    workers = int(args.workers if args.workers is not None else config.WORKERS)
    prefijo = str(args.prefijo_salida or config.PREFIJO_SALIDA)
    guardar = bool(config.GUARDAR_RESULTADOS) and not bool(args.no_guardar)

    inicio = time.perf_counter()
    if modalidad == "secuencial":
        filas = ejecutar_secuencial(datasets, algoritmos, semillas, modalidad, workers)
    else:
        filas = ejecutar_paralelo(datasets, algoritmos, semillas, modalidad, workers)
    duracion = time.perf_counter() - inicio

    detalle = ordenar_resultados(pd.DataFrame(filas), datasets, algoritmos)
    resumen = construir_resumen(detalle, datasets, algoritmos)
    compacto = construir_resumen_compacto(resumen, datasets, algoritmos)
    exitosas = int((detalle["estado"] == "ok").sum()) if "estado" in detalle else 0
    errores = int((detalle["estado"] == "error").sum()) if "estado" in detalle else 0
    resumen_ejecucion = {
        "configuracion": construir_configuracion_salida(datasets, algoritmos, semillas, modalidad, workers, prefijo, guardar),
        "ejecucion": {
            "tiempo_total_segundos": float(duracion),
            "tiempo_total": formato_duracion(duracion),
            "total_ejecuciones": int(len(detalle)),
            "ejecuciones_exitosas": exitosas,
            "ejecuciones_con_error": errores,
        },
    }

    salida = None
    if guardar:
        salida = crear_carpeta_salida(RAIZ_RESULTADOS, prefijo)
        resumen_ejecucion["ejecucion"]["carpeta_salida"] = str(salida)
        guardar_salidas(salida, detalle, resumen, compacto, resumen_ejecucion)

    print()
    if salida is not None:
        print(f"salida: {salida}")
        print(f"detalle: {salida / 'resultados_detalle.csv'}")
        print(f"resumen: {salida / 'resultados_resumen.csv'}")
        print(f"compacto: {salida / 'resultados_resumen_compacto.csv'}")
        print(f"resumen ejecucion: {salida / 'resumen_ejecucion.json'}")
    else:
        print("salida: no se guardaron archivos")
    print(f"tiempo total: {formato_duracion(duracion)}")
    print(f"ejecuciones: {exitosas}/{len(detalle)} exitosas")
    if errores:
        print(f"errores: {errores} (ver resultados_detalle.csv o resumen_ejecucion.json)")
    if not compacto.empty:
        print(compacto.to_string(index=False))


if __name__ == "__main__":
    main()
