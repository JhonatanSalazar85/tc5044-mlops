"""Carga del modelo exportado y transformación de las peticiones en predicciones.

Se separa de `main.py` para poder probar la lógica sin levantar el servidor y
para que la misma clase sirva a otros clientes (por ejemplo, un proceso por
lotes).
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import time

from loguru import logger
import numpy as np
import pandas as pd

from tc5044_mlops.api.esquemas import COLUMNAS_REZAGO, RegistroDemanda
from tc5044_mlops.config import cargar_params, ruta_proyecto
from tc5044_mlops.features import agregar_calendario
from tc5044_mlops.modeling.exportar import cargar_model_info, cargar_modelo_local


def ruta_modelo() -> Path:
    """Carpeta del modelo: variable ``MODELO_DIR`` o ``api.modelo_dir`` de params.yaml."""
    return ruta_proyecto(os.getenv("MODELO_DIR") or cargar_params("api")["modelo_dir"])


class ServicioModelo:
    """Modelo cargado en memoria más su ficha (``model_info.json``)."""

    def __init__(self, modelo_dir: str | Path, zona_horaria: str | None = None):
        self.modelo_dir = Path(modelo_dir)
        inicio = time.perf_counter()
        self.info = cargar_model_info(self.modelo_dir)
        self.modelo = cargar_modelo_local(self.modelo_dir)
        self.columnas = list(self.info["columnas_entrada"])
        self.estaciones = set(self.info.get("estaciones_entrenamiento") or [])
        self.zona_horaria = zona_horaria or cargar_params("api")["zona_horaria"]
        self.cargado_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        logger.success(f"Modelo {self.info['model_uri']} cargado desde {self.modelo_dir} "
                       f"en {time.perf_counter() - inicio:.2f} s")

    # ------------------------------------------------------------------
    def _hora_local(self, fecha: datetime) -> pd.Timestamp:
        ts = pd.Timestamp(fecha)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(self.zona_horaria).tz_localize(None)
        return ts.floor("h")

    def preparar(self, registros: list[RegistroDemanda]) -> tuple[pd.DataFrame, list[str]]:
        """Registros validados -> DataFrame con las columnas y tipos del entrenamiento.

        Los tipos se fijan igual que en `features.py` (float32 para rezagos y
        senos/cosenos, int8 para hora y fin de semana, int64 para la estación):
        así una fila enviada por la API produce exactamente la misma predicción
        que la misma fila en el pipeline por lotes.
        """
        df = pd.DataFrame([r.model_dump() for r in registros])
        df["fecha_hora"] = pd.to_datetime([self._hora_local(f) for f in df["fecha_hora"]])
        df["origen_id"] = df["origen_id"].astype("int64")
        for col in COLUMNAS_REZAGO:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        df = agregar_calendario(df)

        advertencias = []
        if self.estaciones:
            nuevas = sorted(set(df["origen_id"]) - self.estaciones)
            if nuevas:
                advertencias.append(
                    f"Estaciones no vistas en el entrenamiento {nuevas[:20]}: se predicen con "
                    "la codificación media (TargetEncoder) y el centro del espacio PCA.")
        faltantes = {c: int(df[c].isna().sum()) for c in COLUMNAS_REZAGO if df[c].isna().any()}
        if faltantes:
            advertencias.append(f"Rezagos faltantes imputados con la mediana del entrenamiento: "
                                f"{faltantes}.")
        return df[["fecha_hora", *self.columnas]], advertencias

    def predecir(self, registros: list[RegistroDemanda]) -> tuple[pd.DataFrame, list[str]]:
        """Devuelve ``origen_id, fecha_hora, prediccion`` (bicis/h, nunca negativa)."""
        df, advertencias = self.preparar(registros)
        pred = np.clip(self.modelo.predict(df[self.columnas]), 0, None)
        salida = df[["origen_id", "fecha_hora"]].copy()
        salida["prediccion"] = pred.astype(float)
        return salida, advertencias
