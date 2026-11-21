"""Pruebas de integración: el pipeline completo de extremo a extremo con datos sintéticos.

Recorren las mismas etapas que `dvc repro` y el servicio, en el mismo orden:

    CSV mensuales -> consolidar -> limpiar -> features (rejilla estación × hora)
                  -> partición cronológica -> Pipeline de scikit-learn -> predicción
                  -> métricas -> modelo serializado (MLflow) -> predicción idéntica

Los viajes sintéticos (`tc5044_mlops.sintetico.viajes_sinteticos`) traen los mismos
defectos que la fuente real (duplicados exactos y retiros cancelados de menos de un
minuto), así que la prueba valida también que la limpieza los trate como se decidió en
la Fase 1: borrar solo los duplicados exactos y marcar (no borrar) los inválidos.
"""

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from tc5044_mlops import features
from tc5044_mlops.config import cargar_params
from tc5044_mlops.dataset import consolidar_dataset
from tc5044_mlops.features import particion_temporal, tabla_modelado
from tc5044_mlops.limpieza import limpiar
from tc5044_mlops.modeling.evaluate import BaselineEstacional, metricas_regresion
from tc5044_mlops.modeling.exportar import cargar_modelo_local, guardar_modelo
from tc5044_mlops.modeling.models import crear_modelo
from tc5044_mlops.pipeline import columnas_entrada, construir_pipeline
from tc5044_mlops.sintetico import escribir_csv_mensuales, viajes_sinteticos

pytestmark = pytest.mark.integracion

ESTACIONES = 4
DIAS = 35                                    # 2026-03-01 a 2026-04-04
CORTES = {"corte_val": "2026-03-25", "corte_test": "2026-03-30"}
N_DUPLICADOS = 5


@pytest.fixture(scope="module")
def etapas(tmp_path_factory):
    """Ejecuta las etapas de datos una sola vez y comparte los resultados."""
    raiz = tmp_path_factory.mktemp("pipeline")
    viajes = viajes_sinteticos(dias=DIAS, estaciones=ESTACIONES, n_duplicados=N_DUPLICADOS)
    escribir_csv_mensuales(viajes, raiz / "raw" / "viajes")

    interim = consolidar_dataset(raiz / "raw", raiz / "interim" / "viajes.parquet")
    reporte = limpiar(raiz / "interim" / "viajes.parquet", raiz / "raw" / "nomenclatura",
                      raiz / "processed" / "viajes.parquet", raiz / "limpieza.json",
                      cargar_params("limpiar"))
    limpios = pd.read_parquet(raiz / "processed" / "viajes.parquet")
    tabla = tabla_modelado(limpios[limpios["viaje_valido"]], top_estaciones=None)
    tabla["origen_id"] = tabla["origen_id"].astype("int64")
    return {"raiz": raiz, "crudos": viajes, "interim": interim, "reporte": reporte,
            "limpios": limpios, "tabla": tabla}


@pytest.fixture(scope="module")
def modelo(etapas):
    tr, va, te = particion_temporal(etapas["tabla"], **CORTES)
    cols = columnas_entrada()
    pipe = construir_pipeline(crear_modelo("hgb", {"max_iter": 80}, 42), semilla=42)
    pipe.fit(tr[cols], tr["demanda"])
    return {"pipe": pipe, "cols": cols, "tr": tr, "va": va, "te": te}


# --------------------------------------------------------------------------
# Datos: consolidación, limpieza y tabla de modelado
# --------------------------------------------------------------------------
def test_consolidar_conserva_todos_los_viajes_de_los_csv(etapas):
    assert len(etapas["interim"]) == len(etapas["crudos"])
    assert sorted(etapas["interim"]["periodo"].unique()) == ["2026-03", "2026-04"]


def test_limpieza_borra_solo_duplicados_y_marca_los_cancelados(etapas):
    rep, limpios = etapas["reporte"], etapas["limpios"]
    assert rep["duplicados_eliminados"] == N_DUPLICADOS
    assert len(limpios) == len(etapas["crudos"]) - N_DUPLICADOS
    cancelados = limpios[limpios["duracion_min"] < 1]
    assert len(cancelados) > 0 and not cancelados["viaje_valido"].any()


def test_la_rejilla_tiene_todas_las_horas_y_conserva_los_viajes(etapas):
    tabla, limpios = etapas["tabla"], etapas["limpios"]
    horas = tabla["fecha_hora"].nunique()
    assert len(tabla) == ESTACIONES * horas                      # ceros explícitos
    assert tabla["demanda"].sum() == limpios["viaje_valido"].sum()  # ningún viaje se pierde
    assert not tabla.duplicated(["origen_id", "fecha_hora"]).any()


# --------------------------------------------------------------------------
# Modelo: entrenamiento, predicción y métricas
# --------------------------------------------------------------------------
def test_flujo_completo_produce_metricas_validas(modelo):
    pred = modelo["pipe"].predict(modelo["te"][modelo["cols"]])
    assert pred.shape == (len(modelo["te"]),)
    assert np.isfinite(pred).all()
    m = metricas_regresion(modelo["te"]["demanda"], pred)
    assert set(m) == {"mae", "rmse", "wape", "devianza_poisson", "r2"}
    assert all(np.isfinite(v) for v in m.values())


def test_el_modelo_supera_a_la_linea_base_estacional(modelo):
    cols, va = modelo["cols"], modelo["va"]
    base = BaselineEstacional().fit(modelo["tr"][cols], modelo["tr"]["demanda"])
    mae_base = metricas_regresion(va["demanda"], base.predict(va[cols]))["mae"]
    mae_modelo = metricas_regresion(va["demanda"], modelo["pipe"].predict(va[cols]))["mae"]
    assert mae_modelo < mae_base


def test_modelo_serializado_predice_identico(modelo, tmp_path):
    """Guardar con MLflow (skops) y volver a cargar no cambia ni una predicción."""
    destino = guardar_modelo(modelo["pipe"], tmp_path / "modelo", modelo["va"][modelo["cols"]])
    cargado = cargar_modelo_local(destino)
    X = modelo["te"][modelo["cols"]]
    assert np.array_equal(cargado.predict(X), modelo["pipe"].predict(X))


def test_entradas_degradadas_no_rompen_la_prediccion(modelo):
    """Rezagos faltantes (se imputan) y estaciones nuevas (encoder y PCA neutros)."""
    X = modelo["te"][modelo["cols"]].head(24).copy()
    rezagos = [c for c in X.columns if c.startswith("demanda_")]
    X[rezagos] = np.nan
    X["origen_id"] = 999
    pred = modelo["pipe"].predict(X)
    assert np.isfinite(pred).all()


# --------------------------------------------------------------------------
# Interfaz de línea de comandos (la que invoca dvc.yaml)
# --------------------------------------------------------------------------
def test_cli_de_features_escribe_la_tabla(etapas, tmp_path):
    salida = tmp_path / "demanda.parquet"
    resultado = CliRunner().invoke(features.app, [
        "--entrada", str(etapas["raiz"] / "processed" / "viajes.parquet"),
        "--salida", str(salida)])
    assert resultado.exit_code == 0, resultado.output
    tabla = pd.read_parquet(salida)
    assert {"demanda", "demanda_lag_168h", "mes_sin"} <= set(tabla.columns)
