"""Pruebas de la exportación del modelo campeón y de la verificación de reproducibilidad.

Entrenan modelos pequeños con datos sintéticos en una base SQLite temporal
(no tocan el mlflow.db del proyecto), los exportan como lo hará `make exportar`
y comprueban que la carpeta resultante es autocontenida y reproducible.
"""

import json

import mlflow
from mlflow import MlflowClient
import pandas as pd
import pytest

from tc5044_mlops.config import cargar_params
from tc5044_mlops.modeling import exportar, train
from tc5044_mlops.reproducibilidad import reentrenar_y_comparar, verificar_modelo


@pytest.fixture()
def params_prueba(tmp_path, monkeypatch, cortes_sinteticos):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    params = cargar_params()
    params["particion"] = cortes_sinteticos
    params["mlflow"] |= {"experimento": "prueba-exportar", "modelo_registrado": "modelo-prueba"}
    params["exportar"] |= {"muestra_referencia": 200}
    return params


@pytest.fixture()
def exportado(tabla_sintetica, params_prueba, tmp_path):
    resumen = train.ejecutar(tabla=tabla_sintetica, params=params_prueba,
                             modelos=["ridge", "hgb"], n_trials=0,
                             salida_reporte=tmp_path / "m.json", dir_figuras=tmp_path)
    destino = tmp_path / "models" / "champion"
    info = exportar.exportar_desde_registro(params=params_prueba, destino=destino,
                                            tabla=tabla_sintetica)
    return {"info": info, "destino": destino, "resumen": resumen, "params": params_prueba}


def test_la_carpeta_exportada_es_autocontenida(exportado):
    destino = exportado["destino"]
    for archivo in ("MLmodel", "requirements.txt", "model_info.json", "referencia.parquet"):
        assert (destino / archivo).exists(), archivo
    # el paquete viaja con el modelo (code_paths) para cargarlo en el contenedor
    assert (destino / "code" / "tc5044_mlops" / "pipeline.py").exists()
    info = json.loads((destino / "model_info.json").read_text())
    campeon = MlflowClient().get_model_version_by_alias("modelo-prueba", "champion")
    assert info["model_uri"] == f"models:/modelo-prueba/{campeon.version}"
    assert info["run_id"] == campeon.run_id
    assert len(info["sha256_modelo"]) == 64
    assert {"origen_id", "hora", "es_fin_semana"} <= set(info["columnas_entrada"])


def test_metricas_recalculadas_coinciden_con_el_entrenamiento(exportado):
    info, resumen = exportado["info"], exportado["resumen"]
    fila = resumen[resumen["modelo"] == info["modelo"]].iloc[0]
    assert info["metricas_exportacion"]["val_mae"] == pytest.approx(fila["val_mae"], abs=1e-12)
    assert info["metricas_exportacion"]["test_mae"] == pytest.approx(fila["test_mae"], abs=1e-12)
    # las etiquetas del Registry se redondean a 4 decimales
    assert max(info["dif_metricas_vs_registro"].values()) <= 5e-5


def test_la_exportacion_queda_registrada_en_mlflow(exportado):
    exp = MlflowClient().get_experiment_by_name("prueba-exportar")
    corridas = mlflow.search_runs([exp.experiment_id])
    export = corridas[corridas["tags.tipo"] == "exportacion"]
    assert len(export) == 1
    assert export.iloc[0]["tags.sha256_modelo"] == exportado["info"]["sha256_modelo"]


def test_verificar_reproduce_el_golden_set(exportado):
    resultado = verificar_modelo(exportado["destino"])
    assert resultado["comparacion"]["filas"] == 200
    assert resultado["comparacion"]["identicas_bit_a_bit"]


def test_verificar_detecta_un_artefacto_distinto(exportado):
    ruta = exportado["destino"] / "referencia.parquet"
    ref = pd.read_parquet(ruta)
    ref.loc[0, "prediccion_esperada"] += 0.5
    ref.to_parquet(ruta, index=False)
    resultado = verificar_modelo(exportado["destino"])
    assert not resultado["comparacion"]["dentro_tolerancia"]
    assert resultado["comparacion"]["filas_fuera_tolerancia"] == 1


def test_reentrenar_con_la_semilla_reproduce_metricas(exportado, tabla_sintetica):
    resultado = reentrenar_y_comparar(exportado["destino"], tabla=tabla_sintetica,
                                      params=exportado["params"])
    assert resultado["max_dif_metricas"] < 1e-12
    assert resultado["comparacion_golden_set"]["identicas_bit_a_bit"]


def test_modelo_sintetico_para_ci(tmp_path):
    destino = tmp_path / "demo"
    info = exportar.exportar_sintetico(destino=destino)
    assert info["version"] == "0-sintetico"
    assert (destino / "MLmodel").exists()
    assert verificar_modelo(destino)["comparacion"]["identicas_bit_a_bit"]
