"""Regresiones de integridad sin descargas ni archivos Parquet."""
import importlib.util
from pathlib import Path
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'limpiar_mibici.py'
spec = importlib.util.spec_from_file_location('limpiar_mibici', SCRIPT)
limpieza = importlib.util.module_from_spec(spec)
spec.loader.exec_module(limpieza)

class TestDuplicadosExactos(unittest.TestCase):
    def test_elimina_solo_copias_completas(self):
        df = pd.DataFrame({'viaje_id': pd.array([10, 10, 10, 11], dtype='Int64'),
                           'duracion_min': [5.0, 5.0, 8.0, 5.0], 'periodo': ['2025-01'] * 4})
        esperado = df.iloc[[0, 2, 3]].reset_index(drop=True)
        assert_frame_equal(limpieza.eliminar_duplicados_exactos(df), esperado)

    def test_conserva_ids_repetidos_en_periodos_distintos(self):
        df = pd.DataFrame({'viaje_id': [10, 10], 'duracion_min': [5.0, 5.0],
                           'periodo': ['2025-01', '2025-02']})
        assert_frame_equal(limpieza.eliminar_duplicados_exactos(df), df)

    def test_ids_nulos_no_descartan_viajes_distintos(self):
        df = pd.DataFrame({'viaje_id': pd.array([None, None, None], dtype='Int64'),
                           'duracion_min': [5.0, 8.0, 5.0], 'periodo': ['2025-01'] * 3})
        esperado = df.iloc[:2].reset_index(drop=True)
        assert_frame_equal(limpieza.eliminar_duplicados_exactos(df), esperado)

    def test_no_modifica_entrada_y_es_idempotente(self):
        df = pd.DataFrame({'viaje_id': [1, 1], 'duracion_min': [5.0, 5.0]})
        original = df.copy(deep=True)
        resultado = limpieza.eliminar_duplicados_exactos(df)
        assert_frame_equal(df, original)
        assert_frame_equal(limpieza.eliminar_duplicados_exactos(resultado), resultado)

if __name__ == '__main__':
    unittest.main()
