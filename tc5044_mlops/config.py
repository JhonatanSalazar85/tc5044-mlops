"""Configuración central del proyecto.

Define las rutas del esquema Cookiecutter Data Science y carga `params.yaml`,
que es la única fuente de valores configurables (umbrales, hiperparámetros,
semillas, cortes de partición y datos de MLflow). Así ningún módulo deja
valores fijos en el código y cualquier corrida se puede repetir cambiando
solo el archivo de parámetros.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger
import yaml

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PARAMS_PATH = Path(os.getenv("TC5044_PARAMS", PROJ_ROOT / "params.yaml"))


def cargar_params(seccion: str | None = None, ruta: Path | str | None = None) -> Any:
    """Lee `params.yaml` y devuelve el diccionario completo o una sección.

    Parameters
    ----------
    seccion
        Nombre de la sección de primer nivel (por ejemplo ``"modelos"``). Si es
        ``None`` se devuelve todo el archivo.
    ruta
        Ruta alternativa del archivo; por defecto ``PARAMS_PATH`` (que a su vez
        se puede cambiar con la variable de entorno ``TC5044_PARAMS``).
    """
    ruta = Path(ruta) if ruta else PARAMS_PATH
    with open(ruta, encoding="utf-8") as fh:
        params = yaml.safe_load(fh) or {}
    if seccion is None:
        return params
    if seccion not in params:
        raise KeyError(f"La sección '{seccion}' no existe en {ruta}")
    return copy.deepcopy(params[seccion])


def ruta_proyecto(ruta: str | Path) -> Path:
    """Convierte una ruta relativa de `params.yaml` en absoluta respecto a la raíz."""
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else PROJ_ROOT / ruta


def tracking_uri(params: dict | None = None) -> str:
    """URI de MLflow: la variable de entorno tiene prioridad sobre `params.yaml`.

    Una URI ``sqlite:///archivo.db`` relativa se ancla a la raíz del proyecto,
    para que notebooks y scripts escriban en la misma base sin importar desde
    qué carpeta se ejecuten.
    """
    params = params if params is not None else cargar_params()
    uri = os.getenv("MLFLOW_TRACKING_URI") or params["mlflow"]["tracking_uri"]
    prefijo = "sqlite:///"
    if uri.startswith(prefijo) and not uri.startswith(prefijo + "/"):
        uri = prefijo + str(PROJ_ROOT / uri[len(prefijo):])
    return uri


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
