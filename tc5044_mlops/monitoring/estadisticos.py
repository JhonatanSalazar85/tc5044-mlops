"""Estadísticos de drift entre un conjunto de referencia y uno actual.

- **PSI** (Population Stability Index): compara la proporción de observaciones
  por intervalo (cuantiles de la referencia). Regla habitual de la industria:
  < 0.1 estable, 0.1-0.2 cambio moderado, > 0.2 cambio significativo.
- **KS** (Kolmogorov-Smirnov de dos muestras): distancia máxima entre las
  funciones de distribución acumuladas. Con decenas de miles de filas el
  p-valor es ~0 ante cualquier diferencia mínima, así que se usa el
  **estadístico** (tamaño del efecto) con un umbral, no el p-valor.
- **Tasa de nulos**: detecta rupturas del *feature pipeline* que el PSI no ve
  (el PSI se calcula sobre los valores presentes).
- **Inconsistencia de rezagos**: regla de dominio. ``demanda_lag_1h`` de la
  hora *t* debe ser igual a la demanda observada en *t-1* en la misma estación.
  Si no lo es, el error está en el cálculo de variables y no en el mundo real:
  la respuesta correcta es reparar el pipeline, no reentrenar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-6


def psi(referencia, actual, bins: int = 10) -> float:
    """PSI con intervalos por cuantiles de la referencia (ignora nulos)."""
    ref = np.asarray(referencia, dtype=float)
    act = np.asarray(actual, dtype=float)
    ref, act = ref[~np.isnan(ref)], act[~np.isnan(act)]
    if len(ref) == 0 or len(act) == 0:
        return float("nan")
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(cortes) < 2:                      # referencia constante
        return 0.0 if np.allclose(act, ref[0]) else float("inf")
    cortes[0], cortes[-1] = -np.inf, np.inf
    p_ref = np.histogram(ref, cortes)[0] / len(ref)
    p_act = np.histogram(act, cortes)[0] / len(act)
    p_ref, p_act = np.clip(p_ref, EPS, None), np.clip(p_act, EPS, None)
    return float(np.sum((p_act - p_ref) * np.log(p_act / p_ref)))


def ks(referencia, actual) -> tuple[float, float]:
    """Estadístico y p-valor de KS de dos muestras (ignora nulos)."""
    ref = pd.Series(referencia, dtype=float).dropna()
    act = pd.Series(actual, dtype=float).dropna()
    if ref.empty or act.empty:
        return float("nan"), float("nan")
    r = stats.ks_2samp(ref, act)
    return float(r.statistic), float(r.pvalue)


def tasa_nulos(serie) -> float:
    serie = pd.Series(serie)
    return float(serie.isna().mean()) if len(serie) else float("nan")


def estado_columna(psi_valor: float, ks_valor: float, delta_nulos: float, cfg: dict) -> str:
    """'alerta', 'aviso' u 'ok' para una columna según los umbrales de params.yaml."""
    if delta_nulos >= cfg["nulos_alerta"] or (np.isfinite(psi_valor) and
                                              psi_valor >= cfg["psi_alerta"]):
        return "alerta"
    if (np.isfinite(psi_valor) and psi_valor >= cfg["psi_aviso"]) or \
            (np.isfinite(ks_valor) and ks_valor >= cfg["ks_alerta"]):
        return "aviso"
    return "ok"


def resumen_columnas(referencia: pd.DataFrame, actual: pd.DataFrame, columnas: list[str],
                     cfg: dict) -> pd.DataFrame:
    """Una fila por columna: PSI, KS, nulos, medias y estado."""
    filas = []
    for col in columnas:
        p = psi(referencia[col], actual[col], cfg["bins_psi"])
        k, pv = ks(referencia[col], actual[col])
        n_ref, n_act = tasa_nulos(referencia[col]), tasa_nulos(actual[col])
        filas.append({
            "columna": col, "psi": p, "ks": k, "ks_pvalor": pv,
            "nulos_ref": n_ref, "nulos_actual": n_act, "delta_nulos": n_act - n_ref,
            "media_ref": float(pd.to_numeric(referencia[col]).mean()),
            "media_actual": float(pd.to_numeric(actual[col]).mean()),
            "estado": estado_columna(p, k, n_act - n_ref, cfg),
        })
    return pd.DataFrame(filas)


def inconsistencia_rezagos(tabla: pd.DataFrame, col_lag: str = "demanda_lag_1h",
                           objetivo: str = "demanda") -> float:
    """Fracción de filas cuyo rezago de 1 h no coincide con la demanda observada 1 h antes.

    Solo se comparan pares de horas consecutivas de la misma estación con ambos
    valores presentes; los nulos se vigilan aparte con la tasa de nulos.
    """
    d = tabla[["origen_id", "fecha_hora", col_lag, objetivo]].sort_values(
        ["origen_id", "fecha_hora"])
    g = d.groupby("origen_id")
    previa = g[objetivo].shift(1)
    consecutiva = (d["fecha_hora"] - g["fecha_hora"].shift(1)) == pd.Timedelta(hours=1)
    comparable = consecutiva & previa.notna() & d[col_lag].notna()
    if not comparable.any():
        return float("nan")
    distinta = (d.loc[comparable, col_lag].astype(float) - previa[comparable]).abs() > 1e-6
    return float(distinta.mean())
