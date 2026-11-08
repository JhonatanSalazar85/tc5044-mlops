"""Prueba de extremo a extremo: entrenamiento, MLflow Tracking y Model Registry.

Usa una base SQLite temporal, así no toca el mlflow.db del proyecto.
"""

import json

import mlflow
from mlflow import MlflowClient
import pytest

from tc5044_mlops.config import cargar_params
from tc5044_mlops.modeling import predict, train


@pytest.fixture()
def params_prueba(tmp_path, monkeypatch, cortes_sinteticos):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    params = cargar_params()
    params["particion"] = cortes_sinteticos
    params["optuna"] |= {"n_trials": 2, "espacio": {"learning_rate": [0.05, 0.2],
                                                    "max_iter": [20, 40]}}
    params["modelos"]["random_forest"] |= {"n_estimators": 5}
    params["mlflow"] |= {"experimento": "prueba", "modelo_registrado": "modelo-prueba"}
    return params


def test_entrenamiento_registra_corridas_y_modelos(tabla_sintetica, params_prueba, tmp_path):
    reporte = tmp_path / "metricas.json"
    resumen = train.ejecutar(tabla=tabla_sintetica, params=params_prueba,
                             salida_reporte=reporte, dir_figuras=tmp_path)

    # todos los candidatos + el HGB ajustado con Optuna
    assert set(resumen["modelo"]) == {"baseline", "ridge", "poisson", "random_forest",
                                      "hgb", "hgb_optuna"}
    assert resumen["val_mae"].is_monotonic_increasing

    cliente = MlflowClient()
    exp = cliente.get_experiment_by_name("prueba")
    corridas = mlflow.search_runs([exp.experiment_id])
    assert (corridas["tags.tipo"] == "intento_optuna").sum() == 2
    padre = corridas[corridas["tags.tipo"] == "corrida_padre"].iloc[0]
    assert len(padre["tags.dataset_sha256_contenido"]) == 64   # trazabilidad del dataset

    versiones = cliente.search_model_versions("name='modelo-prueba'")
    assert len(versiones) == 5                      # todos menos la línea base
    campeon = cliente.get_model_version_by_alias("modelo-prueba", "champion")
    assert campeon.tags["modelo"] == resumen[resumen["modelo"] != "baseline"].iloc[0]["modelo"]
    assert "val_mae" in campeon.tags and "hiperparametros" in campeon.tags
    assert cliente.get_model_version_by_alias("modelo-prueba", "challenger")

    datos = json.loads(reporte.read_text())
    assert datos["campeon"] == campeon.tags["modelo"]


def test_predict_carga_el_campeon_por_alias(tabla_sintetica, params_prueba, tmp_path):
    train.ejecutar(tabla=tabla_sintetica, params=params_prueba, modelos=["ridge", "hgb"],
                   n_trials=0, salida_reporte=tmp_path / "m.json", dir_figuras=tmp_path)
    modelo = predict.cargar_modelo("champion", params_prueba)
    salida = predict.predecir(tabla_sintetica.tail(48), modelo, params_prueba)
    assert len(salida) == 48 and (salida["prediccion"] >= 0).all()
