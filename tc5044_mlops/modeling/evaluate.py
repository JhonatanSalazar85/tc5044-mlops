"""Métricas de evaluación y línea base para la demanda estación × hora.

Por qué estas métricas
----------------------
El objetivo es un **conteo** con muchos ceros (madrugadas, estaciones poco
usadas). Eso descarta el MAPE (división entre cero) y orienta la selección:

- ``mae``  : error absoluto medio en bicicletas/hora. Es la métrica de
  selección porque se traduce directo a la operación de rebalanceo.
- ``rmse`` : penaliza los errores grandes (picos de hora punta mal previstos).
- ``wape`` : error absoluto ponderado = sum|e| / sum(y). Porcentaje robusto a
  los ceros, útil para comunicar al negocio.
- ``devianza_poisson`` : función de pérdida natural para conteos; compara
  modelos en la escala en que fueron entrenados (pérdida Poisson).
- ``r2``   : proporción de varianza explicada (referencia).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    r2_score,
)
from sklearn.utils.validation import check_is_fitted


def metricas_regresion(y_real, y_pred, prefijo: str = "") -> dict[str, float]:
    """Calcula MAE, RMSE, WAPE, devianza de Poisson y R².

    Las predicciones negativas se recortan a 0 (no existen demandas negativas).
    """
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    suma = y_real.sum()
    m = {
        "mae": mean_absolute_error(y_real, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "wape": float(np.abs(y_real - y_pred).sum() / suma) if suma > 0 else float("nan"),
        "devianza_poisson": mean_poisson_deviance(y_real, np.clip(y_pred, 1e-6, None)),
        "r2": r2_score(y_real, y_pred),
    }
    return {f"{prefijo}{k}": float(v) for k, v in m.items()}


def error_por_hora(y_real, y_pred, horas) -> pd.DataFrame:
    """MAE por hora del día: muestra si el modelo falla en las horas punta."""
    d = pd.DataFrame({"hora": np.asarray(horas), "real": np.asarray(y_real, dtype=float),
                      "pred": np.clip(np.asarray(y_pred, dtype=float), 0, None)})
    d["error_abs"] = (d["real"] - d["pred"]).abs()
    return d.groupby("hora").agg(mae=("error_abs", "mean"), demanda_media=("real", "mean"))


class BaselineEstacional(RegressorMixin, BaseEstimator):
    """Línea base "ingenua estacional": la demanda de la misma hora hace una semana.

    No aprende parámetros; sirve como piso. Un modelo que no supere a esta
    regla no aporta valor sobre lo que ya hace el operador con el histórico.

    Parameters
    ----------
    columna
        Columna con el rezago estacional (``demanda_lag_168h``).
    respaldo
        Columna para las filas sin rezago (primera semana de cada estación).
    """

    def __init__(self, columna: str = "demanda_lag_168h", respaldo: str = "demanda_media_24h"):
        self.columna = columna
        self.respaldo = respaldo

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.media_global_ = float(np.mean(y)) if y is not None else 0.0
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "media_global_")
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        pred = X[self.columna].astype(float)
        if self.respaldo in X:
            pred = pred.fillna(X[self.respaldo].astype(float))
        return pred.fillna(self.media_global_).to_numpy()
