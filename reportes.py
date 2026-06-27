from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from metricas import resumen_cardinalidades

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def crear_carpeta_salida(raiz: Path, prefijo: str) -> Path:
    raiz.mkdir(parents=True, exist_ok=True)
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    salida = raiz / f"{prefijo}_{marca}"
    salida.mkdir(parents=True, exist_ok=False)
    return salida


def ordenar_resultados(df: pd.DataFrame, datasets: tuple[str, ...], algoritmos: tuple[str, ...]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["dataset"] = pd.Categorical(out["dataset"], categories=list(datasets), ordered=True)
    out["algoritmo"] = pd.Categorical(out["algoritmo"], categories=list(algoritmos), ordered=True)
    columnas = [c for c in ["dataset", "algoritmo", "semilla"] if c in out.columns]
    return out.sort_values(columnas).reset_index(drop=True)


def _media(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.mean(vals)) if vals.size else float("nan")


def _std(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.std(vals)) if vals.size else float("nan")


def construir_resumen(detalle: pd.DataFrame, datasets: tuple[str, ...], algoritmos: tuple[str, ...]) -> pd.DataFrame:
    ok = detalle[detalle["estado"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    group_keys = ["dataset", "algoritmo"]
    resumen = (
        ok.groupby(group_keys, dropna=False, observed=False)
        .agg(
            ejecuciones=("dataset", "size"),
            instancias=("instancias", "first"),
            n_eval=("n_eval", "first"),
            features=("features", "first"),
            n_clusters=("n_clusters", "first"),
            clases=("clases", "first"),
            cardinalidad_objetivo=("cardinalidad_objetivo", "first"),
            semilla_inicial=("semilla", "min"),
            semilla_final=("semilla", "max"),
            modalidad=("modalidad", "first"),
            workers=("workers", "first"),
        )
        .reset_index()
    )

    for col in [
        "silhouette",
        "ari",
        "ami",
        "nmi",
        "open_centers",
        "n_clusters_eval",
        "tiempo_algoritmo_s",
        "ram_max_mb",
        "incremento_ram_mb",
    ]:
        agg = (
            ok.groupby(group_keys, dropna=False, observed=False)[col]
            .agg(media=_media, std=_std)
            .reset_index()
            .rename(columns={"media": f"{col}_mean", "std": f"{col}_std"})
        )
        resumen = resumen.merge(agg, on=group_keys, how="left")

    cardinalidades = (
        ok.groupby(group_keys, dropna=False, observed=False)["cluster_sizes"]
        .agg(cardinalidades_finales_q=resumen_cardinalidades)
        .reset_index()
    )
    resumen = resumen.merge(cardinalidades, on=group_keys, how="left")
    return ordenar_resultados(resumen, datasets, algoritmos)


def _fmt(valor: object, decimales: int = 4) -> str:
    if pd.isna(valor):
        return "-"
    return f"{float(valor):.{decimales}f}"


def _partir_lista(valor: object, separador: str, max_por_linea: int) -> str:
    texto = str(valor)
    if len(texto) <= max_por_linea or separador not in texto:
        return texto
    partes = texto.split(separador)
    lineas: list[str] = []
    actual = ""
    for parte in partes:
        candidato = parte if not actual else f"{actual}{separador}{parte}"
        if len(candidato) > max_por_linea and actual:
            lineas.append(actual)
            actual = parte
        else:
            actual = candidato
    if actual:
        lineas.append(actual)
    return "\n".join(lineas)


def _partir_cardinalidad(valor: object) -> str:
    return _partir_lista(valor, "|", max_por_linea=18)


def _partir_cardinalidad_objetivo(valor: object) -> str:
    return _partir_lista(valor, ",", max_por_linea=24)


def construir_resumen_compacto(resumen: pd.DataFrame, datasets: tuple[str, ...], algoritmos: tuple[str, ...]) -> pd.DataFrame:
    if resumen.empty:
        return pd.DataFrame()
    tabla = ordenar_resultados(resumen, datasets, algoritmos).copy()
    tabla["tam"] = tabla.apply(lambda r: f"{int(r['instancias'])}x{int(r['features'])} k={int(r['n_clusters'])}", axis=1)
    tabla["clases"] = tabla["clases"].astype(int).astype(str)
    tabla["card obj"] = tabla["cardinalidad_objetivo"].astype(str)
    tabla["n eval"] = tabla["n_eval"].astype(int).astype(str)
    tabla["exp"] = tabla.apply(lambda r: f"{int(r['ejecuciones'])} | {int(r['semilla_inicial'])}-{int(r['semilla_final'])}", axis=1)
    pares = [
        ("SIL m|std", "silhouette", 4),
        ("ARI m|std", "ari", 4),
        ("AMI m|std", "ami", 4),
        ("NMI m|std", "nmi", 4),
        ("Centros m|std", "open_centers", 2),
        ("Tiempo (s) m|std", "tiempo_algoritmo_s", 4),
        ("RAM max (MB) m|std", "ram_max_mb", 2),
        ("Inc RAM (MB) m|std", "incremento_ram_mb", 2),
    ]
    for nombre, base, dec in pares:
        tabla[nombre] = tabla.apply(lambda r, b=base, d=dec: f"{_fmt(r[f'{b}_mean'], d)}|{_fmt(r[f'{b}_std'], d)}", axis=1)
    tabla["Cuant. card."] = tabla["cardinalidades_finales_q"].fillna("-").astype(str)
    cols = [
        "dataset",
        "algoritmo",
        "tam",
        "clases",
        "card obj",
        "n eval",
        "exp",
        "SIL m|std",
        "ARI m|std",
        "AMI m|std",
        "NMI m|std",
        "Centros m|std",
        "Cuant. card.",
        "Tiempo (s) m|std",
        "RAM max (MB) m|std",
        "Inc RAM (MB) m|std",
    ]
    return tabla[cols].rename(columns={"algoritmo": "alg"}).reset_index(drop=True)


def celdas_destacadas(resumen: pd.DataFrame) -> set[tuple[str, str, str]]:
    destacadas: set[tuple[str, str, str]] = set()
    if resumen.empty or "dataset" not in resumen.columns:
        return destacadas
    for dataset, grupo in resumen.groupby("dataset", observed=False):
        for col, etiqueta in [
            ("silhouette_mean", "SIL m|std"),
            ("ari_mean", "ARI m|std"),
            ("ami_mean", "AMI m|std"),
            ("nmi_mean", "NMI m|std"),
        ]:
            maximo = grupo[col].max()
            for _, row in grupo[np.isclose(grupo[col], maximo, equal_nan=False)].iterrows():
                destacadas.add((str(dataset), str(row["algoritmo"]), etiqueta))
    return destacadas


def generar_imagenes_tabla(compacto: pd.DataFrame, salida: Path, destacadas: set[tuple[str, str, str]]) -> list[Path]:
    if compacto.empty:
        return []
    rutas: list[Path] = []
    filas_por_pagina = 24
    paginas = math.ceil(len(compacto) / filas_por_pagina)
    for pagina in range(paginas):
        pagina_df = compacto.iloc[pagina * filas_por_pagina : (pagina + 1) * filas_por_pagina].copy()
        if "card obj" in pagina_df.columns:
            pagina_df["card obj"] = pagina_df["card obj"].map(_partir_cardinalidad_objetivo)
        if "Cuant. card." in pagina_df.columns:
            pagina_df["Cuant. card."] = pagina_df["Cuant. card."].map(_partir_cardinalidad)
        etiquetas_columnas = [str(col).replace(" ", "\n", 1) if len(str(col)) > 12 else str(col) for col in pagina_df.columns]
        pesos_columnas = {
            "dataset": 1.35,
            "alg": 0.62,
            "tam": 0.88,
            "clases": 0.55,
            "card obj": 1.95,
            "n eval": 0.62,
            "exp": 0.78,
            "SIL m|std": 1.05,
            "ARI m|std": 1.05,
            "AMI m|std": 1.05,
            "NMI m|std": 1.05,
            "Centros m|std": 1.08,
            "Cuant. card.": 1.75,
            "Tiempo (s) m|std": 1.16,
            "RAM max (MB) m|std": 1.18,
            "Inc RAM (MB) m|std": 1.18,
        }
        pesos = [pesos_columnas.get(str(col), 1.0) for col in pagina_df.columns]
        total_pesos = sum(pesos)
        col_widths = [peso / total_pesos for peso in pesos]
        ancho = max(16.0, 1.08 * total_pesos)
        alto = max(1.55, 0.305 * (len(pagina_df) + 1))
        fig, ax = plt.subplots(figsize=(ancho, alto))
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.axis("off")
        table = ax.table(
            cellText=pagina_df.values,
            colLabels=etiquetas_columnas,
            cellLoc="center",
            colLoc="center",
            colWidths=col_widths,
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.2 if len(pagina_df) > 14 else 8.0)
        for (row, _col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#D9EAF7")
                cell.set_text_props(weight="bold", color="#111111")
            else:
                cell.set_facecolor("#F7F9FC" if row % 2 == 0 else "#FFFFFF")
            cell.set_edgecolor("#B7C4D0")
            cell.set_linewidth(0.6)
        col_idx = {name: idx for idx, name in enumerate(pagina_df.columns)}
        for row_pos, (_, row) in enumerate(pagina_df.iterrows(), start=1):
            for etiqueta in ["SIL m|std", "ARI m|std", "AMI m|std", "NMI m|std"]:
                if (str(row["dataset"]), str(row["alg"]), etiqueta) in destacadas:
                    cell = table[(row_pos, col_idx[etiqueta])]
                    cell.set_facecolor("#D8F3DC")
                    cell.set_text_props(weight="bold", color="#0B3D20")
        nombre = "resultados_resumen_compacto.png" if paginas == 1 else f"resultados_resumen_compacto_{pagina + 1:02d}.png"
        ruta = salida / nombre
        fig.savefig(ruta, dpi=190, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        rutas.append(ruta)
    return rutas


def guardar_salidas(
    salida: Path,
    detalle: pd.DataFrame,
    resumen: pd.DataFrame,
    compacto: pd.DataFrame,
    resumen_ejecucion: dict[str, object],
) -> list[str]:
    archivos = []
    detalle.to_csv(salida / "resultados_detalle.csv", index=False)
    archivos.append("resultados_detalle.csv")
    resumen.to_csv(salida / "resultados_resumen.csv", index=False)
    archivos.append("resultados_resumen.csv")
    compacto.to_csv(salida / "resultados_resumen_compacto.csv", index=False)
    archivos.append("resultados_resumen_compacto.csv")
    if not compacto.empty:
        for ruta in generar_imagenes_tabla(compacto, salida, celdas_destacadas(resumen)):
            archivos.append(ruta.name)
    resumen_ejecucion["archivos_generados"] = archivos + ["resumen_ejecucion.json"]
    (salida / "resumen_ejecucion.json").write_text(json.dumps(resumen_ejecucion, indent=2, ensure_ascii=True), encoding="utf-8")
    return resumen_ejecucion["archivos_generados"]
