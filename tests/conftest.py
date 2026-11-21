"""Datos sintéticos compartidos por las pruebas (no requieren descargar MiBici).

Se generan 6 estaciones con dos tipos de uso distintos y 70 días de demanda
horaria con `tc5044_mlops.sintetico`, que aplica las mismas funciones de
`features.py` que se usan con los datos reales; así las pruebas ejercitan el
código de producción. (Hasta la Fase 2 el generador vivía en este archivo; se
movió al paquete para reutilizarlo en la CI y en la prueba del contenedor.)
"""

import pytest

from tc5044_mlops.sintetico import tabla_sintetica as _tabla_sintetica


@pytest.fixture(scope="session")
def tabla_sintetica():
    return _tabla_sintetica()


@pytest.fixture(scope="session")
def cortes_sinteticos() -> dict:
    """Cortes de partición que dejan train/val/test no vacíos en la tabla sintética."""
    return {"corte_val": "2026-04-20", "corte_test": "2026-04-30"}
