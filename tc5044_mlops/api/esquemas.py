"""Esquemas de entrada y salida del servicio (Pydantic v2).

El cliente envía lo que el operador de MiBici conoce **al momento de pedir la
predicción**: la estación, la hora a predecir y la demanda reciente de esa
estación (rezagos y medias móviles). Las variables de calendario (seno/coseno
de hora, día y mes; fin de semana) **no** se piden: el servicio las calcula con
la misma función que se usó al entrenar (`features.agregar_calendario`), así se
evita el *training-serving skew* por una codificación distinta en el cliente.

Validaciones (devuelven HTTP 422 con el detalle del campo):
- campos desconocidos se rechazan (``extra="forbid"``): un nombre mal escrito
  no debe convertirse en silencio en un rezago faltante;
- ``origen_id`` entero positivo; rezagos y medias en [0, 1000] bicis/h;
- entre 1 y ``api.max_registros`` registros por petición.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from tc5044_mlops.config import cargar_params

_API = cargar_params("api")
MAX_REGISTROS: int = int(_API["max_registros"])
MAX_DEMANDA = 1000.0   # límite de sanidad (el máximo real observado es de decenas por hora)

EJEMPLO_REGISTRO = {
    "origen_id": 51,
    "fecha_hora": "2026-08-31T08:00:00",
    "demanda_lag_1h": 18,
    "demanda_lag_2h": 9,
    "demanda_lag_24h": 31,
    "demanda_lag_168h": 28,
    "demanda_media_3h": 10.33,
    "demanda_media_24h": 11.21,
    "demanda_media_168h": 10.64,
}


def _rezago(descripcion: str):
    return Field(None, ge=0, le=MAX_DEMANDA, description=descripcion)


class RegistroDemanda(BaseModel):
    """Una estación en una hora: entrada de una predicción."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EJEMPLO_REGISTRO]})

    origen_id: int = Field(..., gt=0, description="Id de la estación de retiro (catálogo MiBici).")
    fecha_hora: datetime = Field(
        ..., description="Inicio de la hora a predecir, en hora local de Guadalajara. Se trunca "
                         "a la hora; si trae zona horaria se convierte a America/Mexico_City.")
    demanda_lag_1h: float | None = _rezago("Retiros en la hora anterior.")
    demanda_lag_2h: float | None = _rezago("Retiros hace 2 horas.")
    demanda_lag_24h: float | None = _rezago("Retiros a la misma hora del día anterior.")
    demanda_lag_168h: float | None = _rezago("Retiros a la misma hora de la semana anterior.")
    demanda_media_3h: float | None = _rezago("Media de retiros de las 3 horas previas.")
    demanda_media_24h: float | None = _rezago("Media de retiros de las 24 horas previas.")
    demanda_media_168h: float | None = _rezago("Media de retiros de las 168 horas previas.")


COLUMNAS_REZAGO = [c for c in RegistroDemanda.model_fields if c.startswith("demanda_")]


class SolicitudPrediccion(BaseModel):
    """Cuerpo de ``POST /predict``: uno o varios registros (predicción por lote)."""

    model_config = ConfigDict(extra="forbid",
                              json_schema_extra={"examples": [{"registros": [EJEMPLO_REGISTRO]}]})

    registros: list[RegistroDemanda] = Field(..., min_length=1, max_length=MAX_REGISTROS,
                                             description=f"Entre 1 y {MAX_REGISTROS} registros.")


class Prediccion(BaseModel):
    origen_id: int
    fecha_hora: datetime = Field(..., description="Hora predicha (local, truncada a la hora).")
    prediccion: float = Field(..., ge=0, description="Retiros esperados en esa hora (bicis/h).")


class RespuestaPrediccion(BaseModel):
    modelo: str = Field(..., description="Nombre del modelo registrado en MLflow.")
    version_modelo: str = Field(..., description="Versión del Model Registry.")
    model_uri: str = Field(..., description="URI del artefacto: models:/<nombre>/<versión>.")
    version_api: str
    n_registros: int
    predicciones: list[Prediccion]
    advertencias: list[str] = Field(default_factory=list,
                                    description="Avisos no bloqueantes (estación nueva, rezagos "
                                                "imputados).")


class Salud(BaseModel):
    estado: str = Field(..., description="'ok' si el modelo está cargado.")
    modelo_cargado: bool
    version_api: str
    modelo: str | None = None
    version_modelo: str | None = None
    detalle: str | None = None


class InfoModelo(BaseModel):
    nombre: str
    version: str
    alias: str | None = None
    model_uri: str
    run_id: str | None = None
    modelo: str | None = Field(None, description="Candidato ganador (p. ej. hgb_optuna).")
    algoritmo: str | None = None
    metricas_validacion_prueba: dict[str, float] = Field(
        default_factory=dict, description="Métricas recalculadas al exportar.")
    hiperparametros: dict | None = None
    columnas_entrada: list[str]
    sha256_modelo: str | None = None
    git_commit: str | None = None
    exportado_utc: str | None = None
    entorno_exportacion: dict | None = None
