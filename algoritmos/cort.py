from __future__ import annotations

import numpy as np


class CORT:
    _EPSILON = np.finfo(float).tiny
    _ETIQUETA_ERROR: int = -1

    def __init__(
        self,
        k: int,
        cardinalidades: list | np.ndarray,
    ):
        if int(k) <= 0:
            raise ValueError("k must be > 0")
        if len(cardinalidades) != k:
            raise ValueError(
                f"len(cardinalidades)={len(cardinalidades)} must be equal to k={k}"
            )

        self.k = int(k)
        self.cardinalidades = np.asarray(cardinalidades, dtype=np.int64)
        if np.any(self.cardinalidades <= 0):
            raise ValueError("All cardinalidades values must be > 0")

        self.totalPuntos = int(self.cardinalidades.sum())
        self._lambda_denominador = float(np.log1p(max(self.totalPuntos, 1) / self.k))
        self._cardinalidades_float = self.cardinalidades.astype(float)

        self.centroides = np.zeros((self.k, 0), dtype=float)
        self.fundados = np.zeros(self.k, dtype=bool)
        self.cupoRestante = self.cardinalidades.copy()
        self._conteo = np.zeros(self.k, dtype=np.int64)

        self.KFundados = 0
        self.totalProcesados = 0
        self._historial_fundaciones: list[int] = []

        self._delta_min_contador = 0
        self._delta_min_media = 0.0

        self._activo_mask = np.zeros(self.k, dtype=bool)
        self._vacio_mask = np.ones(self.k, dtype=bool)
        self._activos_idx = np.empty(0, dtype=np.int64)
        self._vacios_idx = np.arange(self.k, dtype=np.int64)
        self._activos_sucios = False
        self._vacios_sucios = False

        self._masa_activa_total = 0.0
        self._capacidad_activa_total = 0.0

    @property
    def conteo(self) -> np.ndarray:
        return self._conteo

    @property
    def historialFundaciones(self) -> list[int]:
        return list(self._historial_fundaciones)

    def _asegurar_dimensionalidad(self, punto: np.ndarray) -> None:
        if self.centroides.shape[1] == 0:
            self.centroides = np.zeros((self.k, punto.shape[0]), dtype=float)

    def _indices_activos(self) -> np.ndarray:
        if self._activos_sucios:
            self._activos_idx = np.flatnonzero(self._activo_mask)
            self._activos_sucios = False
        return self._activos_idx

    def _indices_vacios(self) -> np.ndarray:
        if self._vacios_sucios:
            self._vacios_idx = np.flatnonzero(self._vacio_mask)
            self._vacios_sucios = False
        return self._vacios_idx

    def _seleccionar_cluster_vacio_mayor_cardinalidad(self) -> int | None:
        vacios = self._indices_vacios()
        if vacios.size == 0:
            return None

        capacidades = self.cardinalidades[vacios]
        return int(vacios[int(np.argmax(capacidades))])

    def _costo_capacidad_log_vector(
        self,
        cupos_actuales: np.ndarray,
        puntos_restantes: int,
    ) -> np.ndarray:
        if puntos_restantes <= 0:
            return np.zeros(cupos_actuales.shape, dtype=float)

        ratios = cupos_actuales.astype(float) / float(puntos_restantes)
        ratios = np.clip(ratios, self._EPSILON, 1.0)
        return -np.log(ratios)

    def _costo_estructura_log(self, puntos_restantes: int) -> float:
        por_fundar = self.k - self.KFundados
        if por_fundar <= 0:
            return float("inf")

        prob_fundar = float(por_fundar) / float(max(puntos_restantes, 1))
        prob_fundar = float(np.clip(prob_fundar, self._EPSILON, 1.0))
        return -float(np.log(prob_fundar))

    def _referencia_geometrica(self, delta_min: float) -> float:
        if self._delta_min_contador <= 0:
            return max(float(delta_min), self._EPSILON)
        return float(max(self._delta_min_media, self._EPSILON))

    def _lambda_global(self, delta_min: float) -> float:
        referencia = self._referencia_geometrica(delta_min)
        return float(referencia / self._lambda_denominador)

    def _actualizar_delta_min(self, delta_min: float) -> None:
        if not np.isfinite(delta_min) or delta_min < 0.0:
            return

        self._delta_min_contador += 1
        self._delta_min_media += (
            float(delta_min) - self._delta_min_media
        ) / float(self._delta_min_contador)

    def _deltas_activos(self, punto: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        activos = self._indices_activos()
        if activos.size == 0:
            return activos, np.empty(0, dtype=float)

        diferencias = self.centroides[activos] - punto
        deltas = np.sqrt(np.sum(diferencias * diferencias, axis=1))
        return activos, deltas

    def _mejor_asignacion(
        self,
        activos: np.ndarray,
        deltas: np.ndarray,
        lambda_t: float,
        puntos_restantes: int,
    ) -> tuple[int | None, float, float]:
        if activos.size == 0:
            return None, float("inf"), 0.0

        phi_cap = self._costo_capacidad_log_vector(
            self.cupoRestante[activos],
            puntos_restantes,
        )

        if activos.size <= 1:
            phi_load = np.zeros_like(deltas)
        else:
            conteos_activos = self._conteo[activos].astype(float)
            masa_total = self._masa_activa_total
            capacidad_total = self._capacidad_activa_total

            if capacidad_total <= 0.0:
                phi_load = np.zeros_like(deltas)
            else:
                cuota_ideal = self._cardinalidades_float[activos] / max(
                    capacidad_total,
                    self._EPSILON,
                )
                cuota_despues = (conteos_activos + 1.0) / max(
                    masa_total + 1.0,
                    self._EPSILON,
                )
                ratio = cuota_despues / np.maximum(cuota_ideal, self._EPSILON)
                ratio = np.clip(ratio, self._EPSILON, 1.0 / self._EPSILON)
                phi_load = np.log(ratio)

        costos = deltas + lambda_t * (phi_cap + phi_load)
        posicion = int(np.argmin(costos))
        return (
            int(activos[posicion]),
            float(costos[posicion]),
            float(deltas[posicion]),
        )

    def _mejor_fundacion(
        self,
        puntos_restantes: int,
        lambda_t: float,
        indice_ganador: int | None,
        delta_ganador: float,
    ) -> tuple[int | None, float]:
        vacios = self._indices_vacios()
        if vacios.size == 0:
            return None, float("inf")

        costo_base = lambda_t * self._costo_estructura_log(puntos_restantes)
        phi_cap = self._costo_capacidad_log_vector(
            self.cupoRestante[vacios],
            puntos_restantes,
        )

        if indice_ganador is None or indice_ganador < 0 or indice_ganador >= self.k:
            utilidad = np.zeros(vacios.shape, dtype=float)
        else:
            n_g = int(self._conteo[indice_ganador])
            factor_madurez = float(max(n_g - 1, 0)) / float(max(n_g + 1, 1))
            masa_futura = self._cardinalidades_float[vacios] / float(
                max(puntos_restantes, 1)
            )
            utilidad = float(max(delta_ganador, 0.0)) * factor_madurez * masa_futura

        costos = costo_base + lambda_t * phi_cap - utilidad
        posicion = int(np.argmin(costos))
        return int(vacios[posicion]), float(costos[posicion])

    def _fundar_cluster(self, punto: np.ndarray, indice: int) -> int:
        if indice < 0 or indice >= self.k:
            return self._ETIQUETA_ERROR
        if self.fundados[indice] or self.cupoRestante[indice] <= 0:
            return self._ETIQUETA_ERROR

        self.centroides[indice] = punto.copy()
        self.fundados[indice] = True
        self.KFundados += 1
        self.cupoRestante[indice] -= 1
        self._conteo[indice] += 1
        self._historial_fundaciones.append(int(self.totalProcesados))

        self._vacio_mask[indice] = False
        self._vacios_sucios = True

        if self.cupoRestante[indice] > 0:
            self._activo_mask[indice] = True
            self._activos_sucios = True
            self._masa_activa_total += 1.0
            self._capacidad_activa_total += float(self.cardinalidades[indice])

        return int(indice)

    def _asignar_cluster(self, indice: int, punto: np.ndarray) -> int:
        if not self.fundados[indice] or self.cupoRestante[indice] <= 0:
            return self._ETIQUETA_ERROR

        n_actual = int(self._conteo[indice])
        self.centroides[indice] += (punto - self.centroides[indice]) / float(
            n_actual + 1
        )
        self.cupoRestante[indice] -= 1
        self._conteo[indice] += 1

        self._masa_activa_total += 1.0
        if self.cupoRestante[indice] <= 0:
            self._activo_mask[indice] = False
            self._activos_sucios = True
            self._capacidad_activa_total -= float(self.cardinalidades[indice])
            self._masa_activa_total -= float(self._conteo[indice])

        return int(indice)

    def procesarPunto(self, punto) -> int:
        try:
            punto = np.asarray(punto, dtype=float).reshape(-1)
            self._asegurar_dimensionalidad(punto)

            if self.totalProcesados >= self.totalPuntos:
                return self._ETIQUETA_ERROR

            puntos_restantes = self.totalPuntos - self.totalProcesados
            delta_min = None

            if self.KFundados == 0:
                primer_vacio = self._seleccionar_cluster_vacio_mayor_cardinalidad()
                etiqueta = (
                    self._fundar_cluster(punto, int(primer_vacio))
                    if primer_vacio is not None
                    else self._ETIQUETA_ERROR
                )
            else:
                activos, deltas = self._deltas_activos(punto)
                if activos.size == 0:
                    primer_vacio = self._seleccionar_cluster_vacio_mayor_cardinalidad()
                    etiqueta = (
                        self._fundar_cluster(punto, int(primer_vacio))
                        if primer_vacio is not None
                        else self._ETIQUETA_ERROR
                    )
                else:
                    posicion_cercano = int(np.argmin(deltas))
                    indice_cercano = int(activos[posicion_cercano])
                    delta_cercano = float(deltas[posicion_cercano])
                    delta_min = delta_cercano
                    lambda_t = self._lambda_global(delta_min)

                    indice_asignar, costo_asignar, _ = self._mejor_asignacion(
                        activos,
                        deltas,
                        lambda_t,
                        puntos_restantes,
                    )
                    indice_fundar, costo_fundar = self._mejor_fundacion(
                        puntos_restantes,
                        lambda_t,
                        indice_cercano,
                        delta_cercano,
                    )

                    if indice_asignar is None:
                        etiqueta = (
                            self._fundar_cluster(punto, int(indice_fundar))
                            if indice_fundar is not None
                            else self._ETIQUETA_ERROR
                        )
                    elif indice_fundar is not None and costo_fundar < costo_asignar:
                        etiqueta = self._fundar_cluster(punto, int(indice_fundar))
                    else:
                        etiqueta = self._asignar_cluster(int(indice_asignar), punto)

            if etiqueta != self._ETIQUETA_ERROR:
                if delta_min is not None:
                    self._actualizar_delta_min(delta_min)
                self.totalProcesados += 1

            return etiqueta
        except Exception as e:
            print(f"Error al procesar el punto {punto}: {e}")
            return self._ETIQUETA_ERROR
