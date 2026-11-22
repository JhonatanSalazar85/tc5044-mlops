"""Escenarios sintéticos de data drift a partir del conjunto de validación.

No hay drift natural en la ventana 2025-01 a 2026-08 (Fase 1: variaciones
mensuales de ±13 %), así que se simulan cuatro tipos de cambio realistas para
MiBici, cada uno con una causa y una respuesta operativa distintas:

- ``desplazamiento_medias``: rezagos y medias × factor; la demanda real no
  cambia. *Covariate shift* por un error aguas arriba (la ingesta cuenta de más).
- ``features_faltantes``: rezagos de 1-2 h y media de 3 h nulos. Ruptura del
  *feature pipeline* (cae el feed en tiempo real).
- ``cambio_estacional``: menos viajes por la tarde (tormentas de temporada de
  lluvias). Cambio estacional que altera la relación hora -> demanda.
- ``crecimiento_demanda``: más viajes en todas las horas (ciclovías nuevas,
  temporada alta). *Label shift* acompañado de *covariate shift*.

En los dos últimos la demanda se altera **en la serie de conteos** y los
rezagos se recalculan con `features.agregar_rezagos`: el conjunto resultante es
internamente coherente, como lo sería en el mundo real. Los conteos siguen
siendo enteros: las reducciones se aplican por *thinning* binomial y los
aumentos sumando viajes Poisson.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tc5044_mlops.features import agregar_rezagos, particion_temporal

REZAGOS = ["demanda_lag_1h", "demanda_lag_2h", "demanda_lag_24h", "demanda_lag_168h",
           "demanda_media_3h", "demanda_media_24h", "demanda_media_168h"]


def desplazar_medias(df: pd.DataFrame, columnas: list[str], factor: float) -> pd.DataFrame:
    """Multiplica las columnas indicadas (la demanda real queda igual)."""
    out = df.copy()
    for col in columnas:
        out[col] = (out[col].astype(float) * factor).astype("float32")
    return out


def quitar_features(df: pd.DataFrame, columnas: list[str], fraccion: float = 1.0,
                    semilla: int = 42) -> pd.DataFrame:
    """Pone en nulo una fracción de filas de las columnas indicadas."""
    out = df.copy()
    rng = np.random.default_rng(semilla)
    filas = rng.random(len(out)) < fraccion
    for col in columnas:
        out.loc[filas, col] = np.nan
    return out


def escalar_demanda(demanda: pd.Series, factor: np.ndarray | float,
                    rng: np.random.Generator) -> pd.Series:
    """Escala conteos manteniéndolos enteros (binomial si factor < 1, Poisson si > 1)."""
    d = demanda.to_numpy().astype(int)
    f = np.broadcast_to(np.asarray(factor, dtype=float), d.shape)
    menor = f < 1
    nueva = d.copy()
    nueva[menor] = rng.binomial(d[menor], f[menor])
    mayor = f > 1
    nueva[mayor] = d[mayor] + rng.poisson(d[mayor] * (f[mayor] - 1))
    return pd.Series(nueva, index=demanda.index)


def alterar_serie(tabla: pd.DataFrame, factor_por_fila, desde: str, semilla: int = 42,
                  lags=None, ventanas=None) -> pd.DataFrame:
    """Altera la demanda a partir de ``desde`` y recalcula rezagos en toda la tabla."""
    rng = np.random.default_rng(semilla)
    t = tabla.copy()
    mascara = t["fecha_hora"] >= pd.Timestamp(desde)
    factor = factor_por_fila(t.loc[mascara]) if callable(factor_por_fila) else factor_por_fila
    t.loc[mascara, "demanda"] = escalar_demanda(t.loc[mascara, "demanda"], factor, rng)
    kwargs = {k: tuple(v) for k, v in (("lags", lags), ("ventanas", ventanas)) if v}
    return agregar_rezagos(t.drop(columns=[c for c in t.columns if c.startswith(
        ("demanda_lag_", "demanda_media_"))]), **kwargs)


def generar_escenarios(tabla: pd.DataFrame, params: dict) -> dict[str, pd.DataFrame]:
    """Devuelve ``{'referencia': val, 'control': test, <escenario>: df, ...}``."""
    esc = params["drift"]["escenarios"]
    semilla = params["semilla"]
    par = params["particion"]
    _, val, test = particion_temporal(tabla, **par)
    fin_val = pd.Timestamp(par["corte_test"])

    def periodo_val(t: pd.DataFrame) -> pd.DataFrame:
        return t[(t["fecha_hora"] >= pd.Timestamp(par["corte_val"]))
                 & (t["fecha_hora"] < fin_val)].copy()

    h0, h1 = esc["cambio_estacional"]["horas"]
    f_tarde = esc["cambio_estacional"]["factor"]

    def factor_tarde(sub: pd.DataFrame) -> np.ndarray:
        return np.where(sub["fecha_hora"].dt.hour.between(h0, h1), f_tarde, 1.0)

    lags = params["features"]["lags_horas"]
    ventanas = params["features"]["ventanas_moviles"]
    return {
        "referencia": val.copy(),
        "control": test.copy(),
        "desplazamiento_medias": desplazar_medias(
            val, esc["desplazamiento_medias"]["columnas"], esc["desplazamiento_medias"]["factor"]),
        "features_faltantes": quitar_features(
            val, esc["features_faltantes"]["columnas"], esc["features_faltantes"]["fraccion"],
            semilla),
        "cambio_estacional": periodo_val(alterar_serie(
            tabla, factor_tarde, par["corte_val"], semilla, lags, ventanas)),
        "crecimiento_demanda": periodo_val(alterar_serie(
            tabla, esc["crecimiento_demanda"]["factor"], par["corte_val"], semilla, lags,
            ventanas)),
    }


DESCRIPCION = {
    "control": "Prueba real (jul-ago 2026) sin alterar: comprueba que no haya falsas alarmas.",
    "desplazamiento_medias": "Rezagos y medias × factor con la demanda real igual (error en la "
                             "ingesta).",
    "features_faltantes": "Rezagos de 1-2 h y media de 3 h nulos (cae el feed en tiempo real).",
    "cambio_estacional": "Menos retiros por la tarde (tormentas de temporada de lluvias); "
                         "rezagos recalculados.",
    "crecimiento_demanda": "Más retiros en todas las horas (ciclovías / temporada alta); "
                           "rezagos recalculados.",
}
