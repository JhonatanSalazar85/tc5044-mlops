"""Pruebas del monitoreo de drift: estadísticos, escenarios sintéticos y reglas de alerta."""

import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import pytest

from tc5044_mlops.config import cargar_params
from tc5044_mlops.features import particion_temporal
from tc5044_mlops.modeling.models import crear_modelo
from tc5044_mlops.monitoring import drift
from tc5044_mlops.monitoring import estadisticos as est
from tc5044_mlops.monitoring.escenarios import (
    REZAGOS,
    alterar_serie,
    desplazar_medias,
    escalar_demanda,
    generar_escenarios,
)
from tc5044_mlops.pipeline import columnas_entrada, construir_pipeline
from tc5044_mlops.sintetico import tabla_sintetica

CFG = cargar_params("drift")


# --------------------------------------------------------------------------
# Estadísticos
# --------------------------------------------------------------------------
def test_psi_es_casi_cero_sin_cambio_y_grande_con_desplazamiento():
    rng = np.random.default_rng(0)
    ref = rng.poisson(4, 20_000).astype(float)
    assert est.psi(ref, rng.poisson(4, 20_000)) < 0.02
    assert est.psi(ref, rng.poisson(8, 20_000)) > CFG["psi_alerta"]


def test_psi_ignora_nulos_y_maneja_casos_borde():
    assert np.isnan(est.psi([1.0, 2.0], [np.nan, np.nan]))
    assert est.psi([3.0] * 10, [3.0] * 5) == 0.0          # referencia constante sin cambio


def test_ks_detecta_diferencia_de_distribucion():
    rng = np.random.default_rng(1)
    estadistico, _ = est.ks(rng.normal(0, 1, 5000), rng.normal(1, 1, 5000))
    assert estadistico > CFG["ks_alerta"]


@pytest.mark.parametrize("psi_valor, ks_valor, nulos, esperado", [
    (0.01, 0.01, 0.0, "ok"),
    (0.15, 0.01, 0.0, "aviso"),
    (0.01, 0.20, 0.0, "aviso"),
    (0.50, 0.30, 0.0, "alerta"),
    (np.nan, np.nan, 1.0, "alerta"),                    # columna que se quedó vacía
])
def test_estado_columna(psi_valor, ks_valor, nulos, esperado):
    assert est.estado_columna(psi_valor, ks_valor, nulos, CFG) == esperado


def test_rezagos_coherentes_vs_corruptos(tabla_sintetica):
    assert est.inconsistencia_rezagos(tabla_sintetica) == 0.0
    corrupta = desplazar_medias(tabla_sintetica, ["demanda_lag_1h"], 1.5)
    assert est.inconsistencia_rezagos(corrupta) > 0.5


# --------------------------------------------------------------------------
# Escenarios
# --------------------------------------------------------------------------
def test_escalar_demanda_conserva_conteos_enteros():
    rng = np.random.default_rng(2)
    demanda = pd.Series(rng.poisson(5, 50_000))
    menos = escalar_demanda(demanda, 0.5, rng)
    mas = escalar_demanda(demanda, 1.4, rng)
    assert (menos <= demanda).all() and (menos >= 0).all()
    assert menos.mean() == pytest.approx(0.5 * demanda.mean(), rel=0.03)
    assert mas.mean() == pytest.approx(1.4 * demanda.mean(), rel=0.03)
    assert menos.dtype.kind == "i" and mas.dtype.kind == "i"


def test_alterar_serie_recalcula_rezagos_coherentes(tabla_sintetica):
    alterada = alterar_serie(tabla_sintetica, 0.5, "2026-04-01")
    assert est.inconsistencia_rezagos(alterada) == 0.0
    despues = alterada["fecha_hora"] >= "2026-04-01"
    original = tabla_sintetica.set_index(["origen_id", "fecha_hora"])["demanda"]
    nueva = alterada.set_index(["origen_id", "fecha_hora"])["demanda"]
    assert nueva[~despues.to_numpy()].equals(original.loc[nueva[~despues.to_numpy()].index])
    assert alterada.loc[despues, "demanda"].mean() < 0.6 * tabla_sintetica.loc[
        tabla_sintetica["fecha_hora"] >= "2026-04-01", "demanda"].mean()


@pytest.fixture(scope="module")
def mundo():
    """Ocho meses sintéticos (ene-ago 2026) para usar la partición real de params.yaml."""
    params = cargar_params()
    tabla = tabla_sintetica(dias=243, estaciones=8, inicio="2026-01-01")
    tr, _, _ = particion_temporal(tabla, **params["particion"])
    cols = columnas_entrada(params["pipeline"])
    modelo = construir_pipeline(crear_modelo("hgb", {"max_iter": 100}, 42), semilla=42)
    modelo.fit(tr[cols], tr["demanda"])
    return {"params": params, "tabla": tabla, "modelo": modelo}


def test_generar_escenarios(mundo):
    conjuntos = generar_escenarios(mundo["tabla"], mundo["params"])
    assert set(conjuntos) == {"referencia", "control", "desplazamiento_medias",
                              "features_faltantes", "cambio_estacional", "crecimiento_demanda"}
    n = len(conjuntos["referencia"])
    for nombre in ("desplazamiento_medias", "features_faltantes", "cambio_estacional",
                   "crecimiento_demanda"):
        assert len(conjuntos[nombre]) == n, nombre
    faltan = mundo["params"]["drift"]["escenarios"]["features_faltantes"]["columnas"]
    assert conjuntos["features_faltantes"][faltan].isna().all().all()
    assert set(REZAGOS) <= set(conjuntos["cambio_estacional"].columns)


# --------------------------------------------------------------------------
# Reglas de decisión
# --------------------------------------------------------------------------
def _indicadores(**cambios):
    base = {"max_delta_nulos": 0.0, "inconsistencia_rezagos": 0.0, "fraccion_columnas_alerta": 0.0,
            "n_columnas": 7, "n_columnas_alerta": 0, "n_columnas_aviso": 0, "cambio_mae": 0.0,
            "psi_objetivo": 0.0}
    return base | cambios


@pytest.mark.parametrize("cambios, nivel, accion", [
    ({}, "OK", "ninguna"),
    ({"cambio_mae": 0.07}, "AVISO", "vigilar"),
    ({"fraccion_columnas_alerta": 0.5, "n_columnas_alerta": 4}, "AVISO", "monitoreo_reforzado"),
    ({"max_delta_nulos": 1.0, "n_columnas_alerta": 3}, "ALERTA", "revisar_pipeline"),
    ({"inconsistencia_rezagos": 0.9, "cambio_mae": 0.3}, "ALERTA", "revisar_pipeline"),
    ({"cambio_mae": 0.4, "n_columnas_alerta": 2}, "ALERTA", "reentrenar"),
    ({"cambio_mae": 0.4}, "ALERTA", "reentrenar_e_investigar"),
])
def test_decidir_nivel_y_accion(cambios, nivel, accion):
    obtenido_nivel, obtenida_accion, motivos = drift.decidir(_indicadores(**cambios), CFG)
    assert (obtenido_nivel, obtenida_accion) == (nivel, accion)
    assert accion in drift.ACCIONES
    assert (nivel == "OK") == (not motivos)


# --------------------------------------------------------------------------
# Flujo completo
# --------------------------------------------------------------------------
def test_ejecutar_detecta_cada_escenario(mundo, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    resumen = drift.ejecutar(tabla=mundo["tabla"], modelo=mundo["modelo"],
                             params=mundo["params"], salida=tmp_path / "drift",
                             dir_figuras=tmp_path / "fig", con_evidently=True, con_mlflow=True)
    por_escenario = {r["escenario"]: r for r in resumen["escenarios"]}
    assert por_escenario["control"]["nivel"] == "OK"
    assert por_escenario["desplazamiento_medias"]["accion"] == "revisar_pipeline"
    assert por_escenario["features_faltantes"]["accion"] == "revisar_pipeline"
    assert por_escenario["crecimiento_demanda"]["nivel"] == "ALERTA"
    assert por_escenario["crecimiento_demanda"]["cambio_mae"] > CFG["mae_alerta"]

    for archivo in ("resumen_drift.json", "metricas_drift.json", "severidad.csv", "alertas.md",
                    "evidently_desplazamiento_medias.html", "evidently_features_faltantes.html"):
        assert (tmp_path / "drift" / archivo).exists(), archivo
    for figura in ("psi", "desempeno", "severidad", "predicciones"):
        assert (tmp_path / "fig" / f"fase3_drift_{figura}.png").exists()

    severidad = pd.read_csv(tmp_path / "drift" / "severidad.csv")
    assert severidad["psi_max"].is_monotonic_increasing
    assert severidad.loc[severidad["factor"] == 1.0, "nivel"].item() == "OK"

    exp = MlflowClient().get_experiment_by_name(CFG["experimento"])
    corridas = mlflow.search_runs([exp.experiment_id])
    assert (corridas["tags.tipo"] == "monitoreo_drift").sum() == 1
    assert corridas["tags.escenario"].notna().sum() == 5
