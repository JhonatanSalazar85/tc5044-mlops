"""Datos sintéticos compartidos por las pruebas (no requieren descargar MiBici).

Se generan 6 estaciones con dos tipos de uso distintos y 70 días de demanda
horaria, y se les aplican las mismas funciones de `features.py` que a los
datos reales, de modo que las pruebas ejercitan el código de producción.
"""

import numpy as np
import pandas as pd
import pytest

from tc5044_mlops.features import agregar_calendario, agregar_rezagos


def _tabla_sintetica(dias: int = 70, semilla: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    horas = pd.date_range("2026-03-01", periods=24 * dias, freq="h")
    filas = []
    for est in range(1, 7):
        hora = np.asarray(horas.hour)
        finde = np.asarray(horas.day_of_week >= 5)
        if est <= 3:      # estaciones "de trabajo": pico 8 h y 18 h entre semana
            base = 2 + 6 * np.exp(-((hora - 8) ** 2) / 4) + 5 * np.exp(-((hora - 18) ** 2) / 4)
            base = np.where(finde, base * 0.3, base)
        else:             # estaciones "de ocio": pico al mediodía en fin de semana
            base = 1 + 4 * np.exp(-((hora - 13) ** 2) / 9)
            base = np.where(finde, base * 2.0, base)
        filas.append(pd.DataFrame({"origen_id": est, "fecha_hora": horas,
                                   "demanda": rng.poisson(base * (0.5 + est / 6))}))
    dem = pd.concat(filas, ignore_index=True)
    return agregar_rezagos(agregar_calendario(dem))


@pytest.fixture(scope="session")
def tabla_sintetica() -> pd.DataFrame:
    return _tabla_sintetica()


@pytest.fixture(scope="session")
def cortes_sinteticos() -> dict:
    """Cortes de partición que dejan train/val/test no vacíos en la tabla sintética."""
    return {"corte_val": "2026-04-20", "corte_test": "2026-04-30"}
