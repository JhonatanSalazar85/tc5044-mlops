"""
Construcción de la tabla de modelado para la predicción de demanda.

El problema de ML es: dado el histórico, predecir cuántas bicicletas se
retirarán de una estación en una hora determinada. Eso exige pasar de un
registro por viaje a una rejilla estación × hora, incluyendo las horas con
demanda cero — que son información, no ausencia de datos: una estación
vacía a las 3 a.m. es una observación válida y abundante.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from tc5044_mlops.config import cargar_params, ruta_proyecto

# Valores por defecto de las funciones; en el pipeline se leen de params.yaml
# (sección `features`), de modo que cambiar un rezago no exige tocar código.
_PARAMS = cargar_params("features")
LAGS_HORAS = tuple(_PARAMS["lags_horas"])
VENTANAS_MOVILES = tuple(_PARAMS["ventanas_moviles"])

app = typer.Typer(help="Construcción de la tabla de modelado estación × hora.")


def construir_demanda(
    df: pd.DataFrame,
    top_estaciones: int | None = 50,
    incluir_ceros: bool = True,
) -> pd.DataFrame:
    """Agrega los viajes a una rejilla estación × hora.

    Parameters
    ----------
    df
        Dataset limpio (salida de `limpiar_mibici.py`), ya filtrado a
        viajes válidos.
    top_estaciones
        Si se indica, conserva solo las N estaciones con más salidas. La
        rejilla completa son ~472 estaciones × ~14,600 horas ≈ 6.9 M filas;
        acotar a las más activas mantiene el experimento manejable sin
        perder las estaciones que concentran la demanda.
    incluir_ceros
        Reindexa sobre la rejilla completa para que las horas sin viajes
        aparezcan con demanda 0.
    """
    d = df.copy()
    d["fecha_hora"] = pd.to_datetime(d["inicio_viaje"]).dt.floor("h")

    if top_estaciones:
        activas = d["origen_id"].value_counts().head(top_estaciones).index
        d = d[d["origen_id"].isin(activas)]
        logger.info(f"{len(activas)} estaciones, {len(d):,} viajes")

    dem = (d.groupby(["origen_id", "fecha_hora"]).size()
             .rename("demanda").reset_index())

    if incluir_ceros:
        rejilla = pd.MultiIndex.from_product(
            [sorted(dem["origen_id"].unique()),
             pd.date_range(dem["fecha_hora"].min(),
                           dem["fecha_hora"].max(), freq="h")],
            names=["origen_id", "fecha_hora"],
        )
        dem = (dem.set_index(["origen_id", "fecha_hora"])
                  .reindex(rejilla, fill_value=0).reset_index())
        logger.info(f"rejilla completa: {len(dem):,} filas "
                    f"({100 * (dem.demanda == 0).mean():.1f} % en cero)")

    return dem.sort_values(["origen_id", "fecha_hora"]).reset_index(drop=True)


def agregar_calendario(dem: pd.DataFrame) -> pd.DataFrame:
    """Variables de calendario. La hora y el día se codifican además con
    seno/coseno para que el modelo capte su naturaleza cíclica: la hora 23
    está junto a la 0, no a 23 unidades de distancia."""
    d = dem.copy()
    fh = d["fecha_hora"]
    d["hora"] = fh.dt.hour.astype("int8")
    d["dia_semana"] = fh.dt.day_of_week.astype("int8")
    d["es_fin_semana"] = d["dia_semana"].isin([5, 6]).astype("int8")
    d["mes"] = fh.dt.month.astype("int8")
    d["dia_del_anio"] = fh.dt.day_of_year.astype("int16")

    d["hora_sin"] = np.sin(2 * np.pi * d["hora"] / 24).astype("float32")
    d["hora_cos"] = np.cos(2 * np.pi * d["hora"] / 24).astype("float32")
    d["dia_sin"] = np.sin(2 * np.pi * d["dia_semana"] / 7).astype("float32")
    d["dia_cos"] = np.cos(2 * np.pi * d["dia_semana"] / 7).astype("float32")
    d["mes_sin"] = np.sin(2 * np.pi * (d["mes"] - 1) / 12).astype("float32")
    d["mes_cos"] = np.cos(2 * np.pi * (d["mes"] - 1) / 12).astype("float32")
    return d


def agregar_rezagos(dem: pd.DataFrame,
                    lags: tuple[int, ...] = LAGS_HORAS,
                    ventanas: tuple[int, ...] = VENTANAS_MOVILES) -> pd.DataFrame:
    """Rezagos y medias móviles por estación.

    Las medias móviles se calculan con `shift(1)` antes de la ventana para
    no incluir la hora que se está prediciendo: de lo contrario habría fuga
    de información (data leakage) y las métricas offline saldrían
    optimistas de forma irreal.

    El desplazamiento y la ventana se aplican dentro de cada estación con
    `groupby(...).transform`. Aplicar `rolling` sobre la serie ya desplazada
    pero sin agrupar hacía que las primeras horas de cada estación
    promediaran valores del final de la estación anterior, es decir, del
    futuro. Ese defecto se detectó en revisión de código (PR #2) y lo fija
    la prueba `tests/test_features.py`.
    """
    d = dem.sort_values(["origen_id", "fecha_hora"]).copy()
    g = d.groupby("origen_id")["demanda"]

    for lag in lags:
        d[f"demanda_lag_{lag}h"] = g.shift(lag).astype("float32")

    for ventana in ventanas:
        d[f"demanda_media_{ventana}h"] = g.transform(
            lambda s, v=ventana: s.shift(1).rolling(v, min_periods=1).mean()
        ).astype("float32")

    return d


def particion_temporal(dem: pd.DataFrame, corte_val: str, corte_test: str
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partición cronológica, nunca aleatoria.

    Un `train_test_split` al azar sobre series de tiempo entrena con el
    futuro y evalúa con el pasado; el resultado es una métrica inflada que
    no se reproduce en producción.
    """
    fh = dem["fecha_hora"]
    tr = dem[fh < corte_val]
    va = dem[(fh >= corte_val) & (fh < corte_test)]
    te = dem[fh >= corte_test]
    for nom, p in [("train", tr), ("val", va), ("test", te)]:
        if len(p):
            logger.info(f"{nom:6s}: {len(p):>9,} filas  "
                        f"{p.fecha_hora.min():%Y-%m-%d} a {p.fecha_hora.max():%Y-%m-%d}")
    return tr, va, te


def tabla_modelado(df: pd.DataFrame,
                   top_estaciones: int | None = _PARAMS["top_estaciones"],
                   lags: tuple[int, ...] = LAGS_HORAS,
                   ventanas: tuple[int, ...] = VENTANAS_MOVILES) -> pd.DataFrame:
    """Pipeline completo: viajes -> tabla lista para entrenar."""
    return agregar_rezagos(
        agregar_calendario(construir_demanda(df, top_estaciones=top_estaciones)),
        lags=tuple(lags), ventanas=tuple(ventanas))


def cargar_tabla(ruta: str | Path | None = None) -> pd.DataFrame:
    """Lee la tabla de modelado guardada por la etapa `features`."""
    ruta = ruta_proyecto(ruta or _PARAMS["salida"])
    tabla = pd.read_parquet(ruta)
    tabla["fecha_hora"] = pd.to_datetime(tabla["fecha_hora"])
    # El id de estación llega como entero "nullable" (Int64) desde la consolidación;
    # tras la agregación ya no tiene nulos y se pasa a int64 para el pipeline.
    tabla["origen_id"] = tabla["origen_id"].astype("int64")
    return tabla


@app.command()
def main(
    entrada: Path = typer.Option(None, help="Viajes limpios (por defecto params.yaml)"),
    salida: Path = typer.Option(None, help="Tabla de modelado (por defecto params.yaml)"),
) -> None:
    """Viajes limpios -> rejilla estación × hora con calendario y rezagos."""
    par = cargar_params("features")
    entrada = ruta_proyecto(entrada or par["entrada"])
    salida = ruta_proyecto(salida or par["salida"])

    viajes = pd.read_parquet(entrada)
    if "viaje_valido" in viajes.columns:
        viajes = viajes[viajes["viaje_valido"]]
    tabla = tabla_modelado(viajes, top_estaciones=par["top_estaciones"],
                           lags=par["lags_horas"], ventanas=par["ventanas_moviles"])
    salida.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(salida, index=False, compression="zstd")
    logger.success(f"{len(tabla):,} filas x {tabla.shape[1]} columnas -> {salida}")


if __name__ == "__main__":
    app()
