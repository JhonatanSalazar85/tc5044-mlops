"""Pruebas unitarias de reproducibilidad: semillas, huella del entorno y comparación."""

import numpy as np
import pytest

from tc5044_mlops.features import particion_temporal
from tc5044_mlops.modeling.models import crear_modelo
from tc5044_mlops.pipeline import columnas_entrada, construir_pipeline
from tc5044_mlops.reproducibilidad import comparar_predicciones, fijar_semillas, huella_entorno


def test_fijar_semillas_repite_el_generador_global():
    fijar_semillas(42)
    a = np.random.rand(5)
    fijar_semillas(42)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_huella_entorno_registra_python_y_paquetes(monkeypatch):
    monkeypatch.setenv("ENTORNO", "prueba")
    huella = huella_entorno()
    assert huella["entorno"] == "prueba"
    assert huella["python"].startswith("3.11")
    for paquete in ("numpy", "pandas", "scikit-learn", "mlflow"):
        assert paquete in huella["paquetes"]


def test_comparar_predicciones_identicas_y_con_ruido():
    y = np.array([0.0, 1.5, 3.25])
    assert comparar_predicciones(y, y.copy())["identicas_bit_a_bit"]
    casi = comparar_predicciones(y, y + 1e-12, tolerancia=1e-9)
    assert not casi["identicas_bit_a_bit"] and casi["dentro_tolerancia"]
    lejos = comparar_predicciones(y, y + np.array([0, 0, 1e-3]), tolerancia=1e-9)
    assert lejos["filas_fuera_tolerancia"] == 1 and not lejos["dentro_tolerancia"]
    assert lejos["max_abs_diff"] == pytest.approx(1e-3)


def test_comparar_predicciones_rechaza_tamanos_distintos():
    with pytest.raises(ValueError):
        comparar_predicciones([1.0, 2.0], [1.0])


def test_misma_semilla_mismo_modelo(tabla_sintetica, cortes_sinteticos):
    """Dos entrenamientos con la misma semilla dan predicciones idénticas bit a bit.

    El pipeline tiene dos fuentes de aleatoriedad (el barajado del TargetEncoder
    y la partición interna de early stopping del HGB); ambas reciben la semilla.
    """
    tr, va, _ = particion_temporal(tabla_sintetica, **cortes_sinteticos)
    cols = columnas_entrada()

    def entrenar(semilla):
        pipe = construir_pipeline(crear_modelo("hgb", {"max_iter": 40}, semilla), semilla=semilla)
        return pipe.fit(tr[cols], tr["demanda"]).predict(va[cols])

    assert np.array_equal(entrenar(42), entrenar(42))
