"""Utilidades de MLflow: configuración, trazabilidad y Model Registry.

Centraliza todo lo que toca MLflow para que el resto del código no repita la
configuración ni los nombres. Reglas que aplica:

- La URI se toma de ``MLFLOW_TRACKING_URI`` o de ``params.yaml``.
- Cada corrida lleva el **SHA-256 del archivo de datos completo** y el commit
  de Git. En la Fase 1 se comprobó que el *digest* de ``mlflow.data`` solo
  mira las primeras 10 000 filas y no detectó una corrección de los datos;
  el SHA-256 sí cambia con cualquier byte.
- Los modelos se registran en el Model Registry con etiquetas (algoritmo,
  métricas, hiperparámetros) y alias ``champion`` / ``challenger`` en lugar de
  las etapas (Staging/Production), que MLflow 3 ya no recomienda.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess

from loguru import logger
import mlflow
from mlflow import MlflowClient

from tc5044_mlops.config import PROJ_ROOT, cargar_params, tracking_uri

# Tipos que viajan dentro del Pipeline serializado y que skops no confía por
# defecto. MLflow 3 guarda los modelos de scikit-learn con skops, un formato que
# no ejecuta código arbitrario al cargar, y exige declarar cada tipo adicional.
# Se declaran uno por uno (nunca "todo lo que reporte get_untrusted_types") y
# solo porque los archivos los genera y los consume este mismo proyecto.
TIPOS_PROPIOS = [
    "numpy.dtype",  # descriptor de tipo que guardan imputadores y codificadores
    "tc5044_mlops.pipeline.PerfilHorarioPCA",
    "tc5044_mlops.modeling.evaluate.BaselineEstacional",
    "sklearn.tree._tree.Tree",  # árboles del RandomForest
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",  # árboles del HGB
]


# Paquetes que necesita el modelo para cargarse e inferir (no los de desarrollo).
PAQUETES_MODELO = ["mlflow", "scikit-learn", "pandas", "numpy", "scipy", "skops",
                   "loguru", "python-dotenv", "pyyaml", "typer"]


def requisitos_modelo() -> list[str]:
    """Dependencias fijadas con las versiones instaladas.

    Se declaran explícitamente en lugar de dejar que MLflow las infiera: la
    inferencia valida conflictos con ``pip``, que no existe en los entornos
    creados con ``uv``, y además arrastraría dependencias de desarrollo.
    """
    reqs = []
    for paquete in PAQUETES_MODELO:
        try:
            reqs.append(f"{paquete}=={version(paquete)}")
        except PackageNotFoundError:
            continue
    return reqs


def ruta_codigo_paquete() -> str:
    """Carpeta del paquete; se copia dentro del modelo (``code_paths``) para que
    las clases propias (PerfilHorarioPCA, BaselineEstacional) viajen con él y
    el modelo se pueda cargar en otro entorno, por ejemplo un contenedor."""
    return str(PROJ_ROOT / "tc5044_mlops")


def configurar_mlflow(params: dict | None = None) -> str:
    """Fija la URI de tracking y el experimento; devuelve la URI usada.

    Si el proceso corre dentro de ``mlflow run`` (MLflow Projects), el
    experimento ya viene dado por la variable ``MLFLOW_EXPERIMENT_ID`` y no se
    cambia.
    """
    params = params if params is not None else cargar_params()
    uri = tracking_uri(params)
    mlflow.set_tracking_uri(uri)
    if not os.getenv("MLFLOW_RUN_ID") and not os.getenv("MLFLOW_EXPERIMENT_ID"):
        mlflow.set_experiment(params["mlflow"]["experimento"])
    logger.info(f"MLflow tracking: {uri}")
    return uri


def sha256_archivo(ruta: str | Path, bloque: int = 1 << 20) -> str:
    """Hash SHA-256 del archivo completo (no de una muestra)."""
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        while trozo := fh.read(bloque):
            h.update(trozo)
    return h.hexdigest()


def commit_git() -> str:
    """Commit actual del repositorio (o 'sin-git' si no se puede leer)."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJ_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "sin-git"


def sha256_dataframe(df) -> str:
    """Hash SHA-256 del contenido completo de un DataFrame (todas las filas)."""
    import pandas as pd

    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy()).hexdigest()


def etiquetas_trazabilidad(ruta_datos: str | Path | None = None, tabla=None) -> dict[str, str]:
    """Etiquetas comunes a todas las corridas de la fase.

    ``dataset_sha256`` es el hash del archivo Parquet completo y
    ``dataset_sha256_contenido`` el del contenido de la tabla en memoria
    (no depende de la compresión del archivo).
    """
    tags = {"proyecto": "MiBici demanda estación-hora", "equipo": "Equipo 15",
            "fase": "2", "git_commit": commit_git()}
    if ruta_datos and Path(ruta_datos).exists():
        ruta = Path(ruta_datos)
        tags["dataset_ruta"] = str(ruta.relative_to(PROJ_ROOT) if ruta.is_relative_to(PROJ_ROOT)
                                   else ruta)
        tags["dataset_sha256"] = sha256_archivo(ruta)
    if tabla is not None:
        tags["dataset_sha256_contenido"] = sha256_dataframe(tabla)
        tags["dataset_filas"] = str(len(tabla))
    return tags


def hiperparametros_simples(estimador, prefijo: str = "modelo__") -> dict:
    """Hiperparámetros escalares de un estimador (lo que MLflow puede registrar)."""
    return {f"{prefijo}{k}": v for k, v in estimador.get_params(deep=False).items()
            if isinstance(v, (int, float, str, bool)) or v is None}


def registrar_version(model_uri: str, nombre: str, etiquetas: dict,
                      descripcion: str = "") -> int:
    """Registra una versión del modelo y le agrega etiquetas y descripción."""
    version = mlflow.register_model(model_uri, nombre,
                                    tags={k: str(v) for k, v in etiquetas.items()})
    cliente = MlflowClient()
    if descripcion:
        cliente.update_model_version(nombre, version.version, description=descripcion)
    logger.info(f"Model Registry: {nombre} v{version.version} ({etiquetas.get('modelo')})")
    return int(version.version)


def asignar_alias(nombre: str, alias: str, version: int) -> None:
    """Apunta un alias (champion / challenger) a una versión registrada."""
    MlflowClient().set_registered_model_alias(nombre, alias, str(version))
    logger.success(f"Alias @{alias} -> {nombre} v{version}")


def describir_modelo_registrado(nombre: str, descripcion: str) -> None:
    """Crea el modelo registrado si no existe y actualiza su descripción."""
    cliente = MlflowClient()
    try:
        cliente.get_registered_model(nombre)
    except mlflow.exceptions.MlflowException:
        cliente.create_registered_model(nombre, description=descripcion)
        return
    cliente.update_registered_model(nombre, description=descripcion)


def a_json(obj) -> str:
    """Serializa diccionarios con tipos de numpy para etiquetas legibles."""
    return json.dumps(obj, ensure_ascii=False,
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))
