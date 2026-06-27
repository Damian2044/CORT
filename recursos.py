from __future__ import annotations

import threading
import time
import warnings
from typing import Callable, TypeVar

import psutil


T = TypeVar("T")


def ocultar_advertencias() -> None:
    """Oculta advertencias."""
    warnings.filterwarnings("ignore", message="Mean of empty slice.*")
    warnings.filterwarnings("ignore", message="invalid value encountered in divide.*")
    warnings.filterwarnings(
        "ignore",
        message="The number of unique classes is greater than 50% of the number of samples.*",
    )


def medir_tiempo_y_ram(funcion: Callable[[], T], intervalo_segundos: float = 0.01) -> tuple[T, float, float, float]:
    """Mide solo la funcion recibida: tiempo, RAM maxima e incremento de RAM."""
    proceso = psutil.Process()
    ram_inicial = proceso.memory_info().rss
    ram_maxima = ram_inicial
    detener = threading.Event()

    def muestrear() -> None:
        nonlocal ram_maxima
        while not detener.is_set():
            ram_maxima = max(ram_maxima, proceso.memory_info().rss)
            detener.wait(float(intervalo_segundos))

    hilo = threading.Thread(target=muestrear, daemon=True)
    inicio = time.perf_counter()
    hilo.start()
    try:
        resultado = funcion()
    finally:
        ram_maxima = max(ram_maxima, proceso.memory_info().rss)
        detener.set()
        hilo.join()

    tiempo = time.perf_counter() - inicio
    return (
        resultado,
        float(tiempo),
        float(ram_maxima / (1024 * 1024)),
        float((ram_maxima - ram_inicial) / (1024 * 1024)),
    )
