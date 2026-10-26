"""Pruebas de la construcción de variables de demanda sin descargar datos."""
import unittest

import numpy as np
import pandas as pd

from tc5044_mlops.features import agregar_rezagos


def _rejilla(demandas: dict[int, list[int]]) -> pd.DataFrame:
    """Rejilla estación × hora a partir de {estacion: [demanda por hora]}."""
    filas = []
    for est, serie in demandas.items():
        fh = pd.date_range("2025-01-01", periods=len(serie), freq="h")
        filas.append(pd.DataFrame({"origen_id": est, "fecha_hora": fh, "demanda": serie}))
    return pd.concat(filas, ignore_index=True)


class TestRezagosSinFuga(unittest.TestCase):
    def test_medias_moviles_no_cruzan_estaciones(self):
        # La estación 1 tiene demanda 10 y la 2 demanda 0. Si la ventana no se
        # reinicia por estación, la primera hora de la 2 hereda valores de la 1.
        d = agregar_rezagos(_rejilla({1: [10] * 6, 2: [0] * 6}), lags=(1,), ventanas=(3,))
        est2 = d[d.origen_id == 2]["demanda_media_3h"].to_numpy()
        self.assertTrue(np.isnan(est2[0]), "la primera hora no tiene historia previa")
        self.assertTrue((est2[1:] == 0).all(), "no debe arrastrar demanda de la estación 1")

    def test_media_movil_no_incluye_la_hora_actual(self):
        # Un pico en la hora 3 no puede aparecer en la media de la hora 3.
        d = agregar_rezagos(_rejilla({1: [0, 0, 0, 100]}), lags=(1,), ventanas=(3,))
        self.assertEqual(d["demanda_media_3h"].iloc[3], 0.0)

    def test_rezagos_se_reinician_por_estacion(self):
        d = agregar_rezagos(_rejilla({1: [5, 5, 5], 2: [7, 7, 7]}), lags=(1,), ventanas=(3,))
        primera_hora = d.groupby("origen_id").head(1)
        self.assertTrue(primera_hora["demanda_lag_1h"].isna().all())

    def test_un_nulo_por_estacion_en_la_primera_media(self):
        d = agregar_rezagos(_rejilla({e: [1] * 24 for e in range(5)}), lags=(1,), ventanas=(24,))
        self.assertEqual(int(d["demanda_media_24h"].isna().sum()), 5)


if __name__ == "__main__":
    unittest.main()
