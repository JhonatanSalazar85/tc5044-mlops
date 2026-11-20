"""Exportación del modelo campeón a una carpeta autocontenida para el despliegue.

El servicio (FastAPI) y el contenedor no dependen del servidor de MLflow ni de
``mlflow.db``: usan una copia local del modelo registrado. Este módulo la
genera desde el Model Registry y le agrega lo necesario para auditarla:

    models/champion/
    ├── MLmodel, model.skops, ...   <- artefacto de MLflow tal como se registró
    ├── requirements.txt            <- versiones exactas del entorno de entrenamiento
    ├── code/tc5044_mlops/          <- paquete con las clases propias (code_paths)
    ├── model_info.json             <- nombre, versión, alias, URI, run_id, métricas,
    │                                  hiperparámetros, SHA-256 y huella del entorno
    └── referencia.parquet          <- "golden set": filas de validación con la
                                       predicción esperada, para verificar la
                                       reproducibilidad en otro entorno

Las métricas se **recalculan** con el artefacto descargado sobre validación y
prueba completas y se comparan con las que se guardaron en el Model Registry
al entrenar: es la primera verificación de que el modelo serializado se
comporta igual que el que estaba en memoria.

Uso:
    python -m tc5044_mlops.modeling.exportar                     # @champion -> models/champion
    python -m tc5044_mlops.modeling.exportar --alias challenger --destino models/challenger
    python -m tc5044_mlops.modeling.exportar --sintetico         # modelo demo (CI, sin datos)
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import typer

from tc5044_mlops.config import cargar_params, ruta_proyecto
from tc5044_mlops.features import cargar_tabla, particion_temporal
from tc5044_mlops.modeling.evaluate import metricas_regresion
from tc5044_mlops.pipeline import columnas_entrada
from tc5044_mlops.reproducibilidad import fijar_semillas, huella_entorno
from tc5044_mlops.tracking import (
    TIPOS_PROPIOS,
    a_json,
    commit_git,
    configurar_mlflow,
    requisitos_modelo,
    ruta_codigo_paquete,
    sha256_archivo,
    sha256_dataframe,
)

app = typer.Typer(help="Exporta el modelo del Model Registry a una carpeta autocontenida.")

ARCHIVO_INFO = "model_info.json"
ARCHIVO_REFERENCIA = "referencia.parquet"


# --------------------------------------------------------------------------
# Lectura y escritura de la carpeta del modelo
# --------------------------------------------------------------------------
def cargar_model_info(modelo_dir: str | Path) -> dict:
    """Lee ``model_info.json`` de la carpeta del modelo."""
    ruta = Path(modelo_dir) / ARCHIVO_INFO
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}; corre `make exportar` primero")
    return json.loads(ruta.read_text(encoding="utf-8"))


def cargar_modelo_local(modelo_dir: str | Path):
    """Carga el `Pipeline` de scikit-learn desde la carpeta exportada (sin servidor MLflow)."""
    return mlflow.sklearn.load_model(str(modelo_dir))


def escribir_model_info(modelo_dir: str | Path, info: dict) -> Path:
    ruta = Path(modelo_dir) / ARCHIVO_INFO
    ruta.write_text(json.dumps(json.loads(a_json(info)), indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return ruta


def sha256_modelo(modelo_dir: str | Path) -> str:
    """SHA-256 del archivo serializado del modelo (``model.skops`` o ``model.pkl``)."""
    modelo_dir = Path(modelo_dir)
    for nombre in ("model.skops", "model.pkl"):
        if (modelo_dir / nombre).exists():
            return sha256_archivo(modelo_dir / nombre)
    candidatos = sorted(p for p in modelo_dir.iterdir() if p.name.startswith("model."))
    return sha256_archivo(candidatos[0]) if candidatos else ""


def guardar_modelo(pipe, destino: str | Path, X_ejemplo: pd.DataFrame) -> Path:
    """Guarda un `Pipeline` en formato MLflow con las mismas opciones que ``train.py``.

    Se usa para el modelo sintético de demostración y en las pruebas; el
    modelo real se descarga del Registry con :func:`descargar_modelo`.
    """
    destino = Path(destino)
    if destino.exists():
        shutil.rmtree(destino)
    firma = infer_signature(X_ejemplo.head(200), pipe.predict(X_ejemplo.head(200)))
    mlflow.sklearn.save_model(pipe, str(destino), signature=firma,
                              input_example=X_ejemplo.head(5),
                              skops_trusted_types=TIPOS_PROPIOS,
                              pip_requirements=requisitos_modelo(),
                              code_paths=[ruta_codigo_paquete()])
    return destino


def descargar_modelo(model_uri: str, destino: str | Path) -> Path:
    """Descarga los artefactos de ``model_uri`` (``models:/<nombre>/<versión>``) a ``destino``."""
    destino = Path(destino)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=tmp))
        if destino.exists():
            shutil.rmtree(destino)
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local, destino)
    logger.info(f"{model_uri} -> {destino}")
    return destino


# --------------------------------------------------------------------------
# Datos de validación y golden set
# --------------------------------------------------------------------------
def particionar(tabla: pd.DataFrame, params: dict) -> dict:
    """Partición cronológica y X / y (igual que ``train.preparar_datos``).

    Se repite aquí para no importar ``train.py``, que arrastra Optuna y
    matplotlib; así este módulo se puede usar dentro del contenedor.
    """
    cols = columnas_entrada(params["pipeline"])
    objetivo = params["pipeline"]["objetivo"]
    tr, va, te = particion_temporal(tabla, **params["particion"])
    return {"X_tr": tr[cols], "y_tr": tr[objetivo], "X_va": va[cols], "y_va": va[objetivo],
            "X_te": te[cols], "y_te": te[objetivo], "val": va, "test": te}


def construir_referencia(modelo, val: pd.DataFrame, params: dict, n: int) -> pd.DataFrame:
    """Muestra fija de validación con sus entradas, la demanda real y la predicción esperada."""
    cols = columnas_entrada(params["pipeline"])
    objetivo = params["pipeline"]["objetivo"]
    n = min(int(n), len(val))
    muestra = val.sample(n=n, random_state=params["semilla"]) if n < len(val) else val
    muestra = muestra.sort_values(["origen_id", "fecha_hora"]).reset_index(drop=True)
    ref = muestra[list(dict.fromkeys(["origen_id", "fecha_hora", *cols, objetivo]))].copy()
    ref["prediccion_esperada"] = np.clip(modelo.predict(ref[cols]), 0, None)
    return ref


def completar_exportacion(destino: Path, info: dict, tabla: pd.DataFrame, params: dict,
                          ruta_tabla: Path | None = None) -> dict:
    """Recalcula métricas, guarda el golden set y escribe ``model_info.json``."""
    modelo = cargar_modelo_local(destino)
    datos = particionar(tabla, params)
    metricas = (metricas_regresion(datos["y_va"], modelo.predict(datos["X_va"]), "val_")
                | metricas_regresion(datos["y_te"], modelo.predict(datos["X_te"]), "test_"))

    ref = construir_referencia(modelo, datos["val"], params,
                               params["exportar"]["muestra_referencia"])
    ruta_ref = destino / ARCHIVO_REFERENCIA
    ref.to_parquet(ruta_ref, index=False)

    registro = info.get("metricas_registro") or {}
    diferencias = {k: abs(float(v) - metricas[k]) for k, v in registro.items() if k in metricas}
    info |= {
        "columnas_entrada": columnas_entrada(params["pipeline"]),
        "objetivo": params["pipeline"]["objetivo"],
        "metricas_exportacion": metricas,
        "dif_metricas_vs_registro": diferencias,
        "sha256_modelo": sha256_modelo(destino),
        "referencia": {
            "archivo": ARCHIVO_REFERENCIA,
            "filas": len(ref),
            "sha256": sha256_archivo(ruta_ref),
            "metricas": metricas_regresion(ref[params["pipeline"]["objetivo"]],
                                           ref["prediccion_esperada"]),
        },
        "dataset": {
            "ruta": str(ruta_tabla) if ruta_tabla else None,
            "sha256_archivo": (sha256_archivo(ruta_tabla)
                               if ruta_tabla and Path(ruta_tabla).exists() else None),
            "sha256_contenido": sha256_dataframe(tabla),
            "filas": len(tabla),
            "particion": params["particion"],
        },
        "estaciones_entrenamiento": sorted(int(e) for e in datos["X_tr"]["origen_id"].unique()),
        "git_commit": commit_git(),
        "semilla": params["semilla"],
        "exportado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entorno_exportacion": huella_entorno(),
    }
    escribir_model_info(destino, info)
    logger.success(f"Exportado {info['model_uri']} -> {destino} "
                   f"(val_mae={metricas['val_mae']:.4f}, referencia={len(ref)} filas)")
    if diferencias and max(diferencias.values()) > 1e-4:
        logger.warning(f"Las métricas recalculadas difieren de las del Registry: {diferencias}")
    return info


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------
def exportar_desde_registro(params: dict | None = None, alias: str | None = None,
                            destino: str | Path | None = None, tabla: pd.DataFrame | None = None,
                            ruta_tabla: str | Path | None = None,
                            registrar_corrida: bool = True) -> dict:
    """Descarga ``models:/<nombre>@<alias>`` y lo deja listo para el servicio."""
    params = params or cargar_params()
    fijar_semillas(params["semilla"])
    alias = alias or params["exportar"]["alias"]
    destino = ruta_proyecto(destino or params["exportar"]["destino"])
    ruta_tabla = ruta_proyecto(ruta_tabla or params["features"]["salida"])
    nombre = params["mlflow"]["modelo_registrado"]

    configurar_mlflow(params)
    version = MlflowClient().get_model_version_by_alias(nombre, alias)
    model_uri = f"models:/{nombre}/{version.version}"
    descargar_modelo(model_uri, destino)

    tags = dict(version.tags)
    registro = {k: float(tags[k]) for k in ("val_mae", "val_rmse", "val_wape", "test_mae")
                if k in tags}
    info = {
        "nombre": nombre, "version": str(version.version), "alias": alias,
        "model_uri": model_uri, "model_uri_alias": f"models:/{nombre}@{alias}",
        "run_id": version.run_id, "modelo": tags.get("modelo"),
        "algoritmo": tags.get("algoritmo"),
        "hiperparametros": json.loads(tags["hiperparametros"]) if "hiperparametros" in tags
        else None,
        "metricas_registro": registro,
    }
    tabla = cargar_tabla(ruta_tabla) if tabla is None else tabla
    info = completar_exportacion(destino, info, tabla, params, ruta_tabla)

    if registrar_corrida:
        with mlflow.start_run(run_name=f"fase3-exportacion-v{version.version}"):
            mlflow.set_tags({"tipo": "exportacion", "fase": "3", "modelo": info["modelo"],
                             "model_uri": model_uri, "git_commit": info["git_commit"],
                             "dataset_sha256_contenido": info["dataset"]["sha256_contenido"],
                             "sha256_modelo": info["sha256_modelo"]})
            mlflow.log_metrics({f"export_{k}": v for k, v in info["metricas_exportacion"].items()})
            mlflow.log_artifact(str(destino / ARCHIVO_INFO), "exportacion")
            mlflow.log_artifact(str(destino / ARCHIVO_REFERENCIA), "exportacion")
            ruta_ref = destino / ARCHIVO_REFERENCIA
            mlflow.log_input(mlflow.data.from_pandas(
                pd.read_parquet(ruta_ref), source=str(ruta_ref), name="golden-set-referencia",
                targets=params["pipeline"]["objetivo"]), context="referencia")
    return info


def exportar_sintetico(params: dict | None = None, destino: str | Path | None = None) -> dict:
    """Entrena un HGB pequeño con datos sintéticos y lo exporta con el mismo formato.

    Sirve para la integración continua y para probar el contenedor sin los
    datos reales. **No** es el modelo de producción.
    """
    from tc5044_mlops.modeling.models import crear_modelo
    from tc5044_mlops.pipeline import construir_pipeline
    from tc5044_mlops.sintetico import tabla_sintetica

    params = params or cargar_params()
    params["particion"] = {"corte_val": "2026-04-20", "corte_test": "2026-04-30"}
    fijar_semillas(params["semilla"])
    destino = ruta_proyecto(destino or params["exportar"]["destino"])
    tabla = tabla_sintetica()
    datos = particionar(tabla, params)
    pipe = construir_pipeline(crear_modelo("hgb", {"max_iter": 60}, params["semilla"]),
                              cfg=params["pipeline"], semilla=params["semilla"])
    pipe.fit(datos["X_tr"], datos["y_tr"])
    guardar_modelo(pipe, destino, datos["X_va"])
    info = {"nombre": f"{params['mlflow']['modelo_registrado']}-demo", "version": "0-sintetico",
            "alias": "demo", "model_uri": "local:sintetico", "run_id": None,
            "modelo": "hgb_demo", "algoritmo": "hgb",
            "hiperparametros": {"max_iter": 60}, "metricas_registro": {}}
    return completar_exportacion(destino, info, tabla, params)


@app.command()
def main(
    alias: str = typer.Option(None, help="Alias del Model Registry (por defecto params.yaml)"),
    destino: Path = typer.Option(None, help="Carpeta de salida (por defecto params.yaml)"),
    sintetico: bool = typer.Option(False, help="Modelo demo con datos sintéticos (CI)"),
    sin_corrida: bool = typer.Option(False, help="No registrar la corrida de exportación"),
) -> None:
    """Exporta el modelo del alias indicado (o uno sintético) a una carpeta autocontenida."""
    if sintetico:
        info = exportar_sintetico(destino=destino)
    else:
        info = exportar_desde_registro(alias=alias, destino=destino,
                                       registrar_corrida=not sin_corrida)
    print(json.dumps({k: info[k] for k in ("model_uri", "modelo", "sha256_modelo")} |
                     {"val_mae": info["metricas_exportacion"]["val_mae"],
                      "test_mae": info["metricas_exportacion"]["test_mae"]}, indent=2))


if __name__ == "__main__":
    app()
