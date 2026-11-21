"""Pruebas del servicio FastAPI (unitarias de validación y de integración con el modelo).

Se exporta un modelo sintético con el mismo formato que `make exportar`
(`exportar.exportar_sintetico`) y se levanta la app con `TestClient`, que
ejecuta el `lifespan` igual que uvicorn.
"""

import json

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from tc5044_mlops.api.esquemas import COLUMNAS_REZAGO, MAX_REGISTROS
from tc5044_mlops.api.main import crear_app
from tc5044_mlops.modeling import exportar

REGISTRO = {"origen_id": 1, "fecha_hora": "2026-04-25T08:00:00", "demanda_lag_1h": 5,
            "demanda_lag_2h": 3, "demanda_lag_24h": 7, "demanda_lag_168h": 6,
            "demanda_media_3h": 4.0, "demanda_media_24h": 3.5, "demanda_media_168h": 3.2}


@pytest.fixture(scope="module")
def modelo_dir(tmp_path_factory):
    destino = tmp_path_factory.mktemp("modelo") / "champion"
    exportar.exportar_sintetico(destino=destino)
    return destino


@pytest.fixture(scope="module")
def cliente(modelo_dir):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MODELO_DIR", str(modelo_dir))
        with TestClient(crear_app()) as c:
            yield c


def _como_registros(ref: pd.DataFrame) -> list[dict]:
    """Filas del golden set -> cuerpo JSON (NaN -> null, fechas ISO)."""
    regs = ref[["origen_id", "fecha_hora", *COLUMNAS_REZAGO]].copy()
    regs["fecha_hora"] = regs["fecha_hora"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return regs.astype(object).where(regs.notna(), None).to_dict(orient="records")


# --------------------------------------------------------------------------
# Operación y ficha del modelo
# --------------------------------------------------------------------------
def test_health_con_modelo_cargado(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json()["estado"] == "ok" and r.json()["modelo_cargado"]


def test_health_responde_503_sin_modelo(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELO_DIR", str(tmp_path / "no-existe"))
    with TestClient(crear_app()) as c:      # instancia aparte: no afecta al cliente del módulo
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json={"registros": [REGISTRO]}).status_code == 503


def test_model_info_expone_uri_version_y_metricas(cliente, modelo_dir):
    info = cliente.get("/model-info").json()
    esperado = json.loads((modelo_dir / "model_info.json").read_text())
    assert info["model_uri"] == esperado["model_uri"]
    assert info["version"] == esperado["version"]
    assert info["sha256_modelo"] == esperado["sha256_modelo"]
    assert "val_mae" in info["metricas_validacion_prueba"]
    assert "demanda_lag_1h" in info["columnas_entrada"]


# --------------------------------------------------------------------------
# Predicción
# --------------------------------------------------------------------------
def test_predict_un_registro(cliente):
    r = cliente.post("/predict", json={"registros": [REGISTRO]})
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["n_registros"] == 1
    assert cuerpo["predicciones"][0]["prediccion"] >= 0
    assert cuerpo["predicciones"][0]["fecha_hora"] == "2026-04-25T08:00:00"
    assert "x-tiempo-proceso-ms" in r.headers


def test_predict_es_identico_al_pipeline_por_lotes(cliente, modelo_dir):
    """Paridad entrenamiento-servicio: la API reproduce bit a bit el golden set."""
    ref = pd.read_parquet(modelo_dir / "referencia.parquet").head(500)
    r = cliente.post("/predict", json={"registros": _como_registros(ref)})
    assert r.status_code == 200
    pred = np.array([p["prediccion"] for p in r.json()["predicciones"]])
    assert np.array_equal(pred, ref["prediccion_esperada"].to_numpy())


def test_hora_con_zona_horaria_se_convierte_a_hora_local(cliente):
    local = cliente.post("/predict", json={"registros": [REGISTRO]}).json()
    utc = dict(REGISTRO, fecha_hora="2026-04-25T14:00:00Z")      # 08:00 en Guadalajara
    convertida = cliente.post("/predict", json={"registros": [utc]}).json()
    assert convertida["predicciones"][0]["fecha_hora"] == "2026-04-25T08:00:00"
    assert convertida["predicciones"][0]["prediccion"] == local["predicciones"][0]["prediccion"]


def test_minutos_se_truncan_a_la_hora(cliente):
    r = cliente.post("/predict", json={"registros": [dict(REGISTRO,
                                                          fecha_hora="2026-04-25T08:47:00")]})
    assert r.json()["predicciones"][0]["fecha_hora"] == "2026-04-25T08:00:00"


def test_rezagos_faltantes_se_imputan_con_advertencia(cliente):
    registro = {"origen_id": 2, "fecha_hora": "2026-04-25T13:00:00", "demanda_lag_24h": 4}
    r = cliente.post("/predict", json={"registros": [registro]})
    assert r.status_code == 200
    assert any("imputados" in a for a in r.json()["advertencias"])


def test_estacion_no_vista_genera_advertencia(cliente):
    r = cliente.post("/predict", json={"registros": [dict(REGISTRO, origen_id=999)]})
    assert r.status_code == 200
    assert any("999" in a for a in r.json()["advertencias"])


# --------------------------------------------------------------------------
# Validación de entrada (Pydantic -> HTTP 422)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cambio", [
    {"origen_id": 0},                      # estación inválida
    {"origen_id": "doce"},                 # tipo incorrecto
    {"demanda_lag_1h": -1},                # demanda negativa
    {"demanda_lag_24h": 5000},             # fuera del límite de sanidad
    {"fecha_hora": "ayer"},                # fecha ilegible
    {"demanda_lag_1": 3},                  # campo mal escrito (extra="forbid")
])
def test_entradas_invalidas_devuelven_422(cliente, cambio):
    r = cliente.post("/predict", json={"registros": [dict(REGISTRO, **cambio)]})
    assert r.status_code == 422


def test_falta_un_campo_obligatorio(cliente):
    sin_fecha = {k: v for k, v in REGISTRO.items() if k != "fecha_hora"}
    r = cliente.post("/predict", json={"registros": [sin_fecha]})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][-1] == "fecha_hora"


@pytest.mark.parametrize("n", [0, MAX_REGISTROS + 1])
def test_numero_de_registros_fuera_de_rango(cliente, n):
    assert cliente.post("/predict", json={"registros": [REGISTRO] * n}).status_code == 422


# --------------------------------------------------------------------------
# Documentación (OpenAPI)
# --------------------------------------------------------------------------
def test_openapi_documenta_el_esquema(cliente):
    esquema = cliente.get("/openapi.json").json()
    assert {"/predict", "/health", "/model-info"} <= set(esquema["paths"])
    registro = esquema["components"]["schemas"]["RegistroDemanda"]
    assert registro["properties"]["origen_id"]["description"]
    assert "fecha_hora" in registro["required"]
