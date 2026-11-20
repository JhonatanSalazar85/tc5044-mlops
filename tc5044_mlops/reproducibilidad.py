"""Reproducibilidad: semillas, huella del entorno y verificación del modelo desplegable.

La rúbrica de la Fase 3 pide demostrar que **el mismo artefacto produce los
mismos resultados en un entorno distinto al de entrenamiento**. Este módulo
hace dos verificaciones independientes, pensadas para correr en el WSL de
entrenamiento y dentro del contenedor:

1. ``verificar``: carga el modelo exportado (``models/champion``) y predice
   el *golden set* (``referencia.parquet``, guardado junto al modelo por
   ``modeling/exportar.py``). Compara fila por fila con la predicción
   esperada y reporta la diferencia máxima.
2. ``reentrenar``: clona el `Pipeline` del modelo (mismos hiperparámetros y
   mismas semillas), lo vuelve a entrenar con la partición de
   ``params.yaml`` y compara las métricas y las predicciones con las del
   modelo original. Si coinciden, el entrenamiento es reproducible y no solo
   la inferencia.

Uso:
    python -m tc5044_mlops.reproducibilidad verificar
    python -m tc5044_mlops.reproducibilidad verificar --salida -     # JSON a stdout
    python -m tc5044_mlops.reproducibilidad reentrenar
    python -m tc5044_mlops.reproducibilidad huella
"""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

from loguru import logger
import numpy as np
import pandas as pd
import typer

from tc5044_mlops.config import REPORTS_DIR, cargar_params, ruta_proyecto

app = typer.Typer(help="Verificaciones de reproducibilidad del modelo desplegable.")

PAQUETES_HUELLA = ["numpy", "pandas", "scikit-learn", "scipy", "mlflow", "mlflow-skinny",
                   "skops", "fastapi", "pydantic", "uvicorn", "evidently"]


def fijar_semillas(semilla: int | None = None) -> int:
    """Fija las semillas de ``random`` y ``numpy`` y la variable ``PYTHONHASHSEED``.

    Los estimadores de scikit-learn, el ``TargetEncoder`` y el muestreador
    TPE de Optuna reciben además la semilla de forma explícita
    (``random_state`` / ``seed``) desde ``params.yaml``; esta función cubre el
    código que use el generador global. ``PYTHONHASHSEED`` solo afecta a los
    procesos hijos (el actual ya arrancó con su valor).
    """
    semilla = cargar_params("semilla") if semilla is None else int(semilla)
    random.seed(semilla)
    np.random.seed(semilla)
    os.environ["PYTHONHASHSEED"] = str(semilla)
    return semilla


def huella_entorno() -> dict:
    """Versión de Python, sistema y paquetes clave del entorno actual.

    La etiqueta ``ENTORNO`` (variable de entorno) distingue, por ejemplo,
    ``wsl-entrenamiento`` de ``contenedor``.
    """
    paquetes = {}
    for nombre in PAQUETES_HUELLA:
        try:
            paquetes[nombre] = version(nombre)
        except PackageNotFoundError:
            continue
    return {
        "entorno": os.getenv("ENTORNO", "local"),
        "python": platform.python_version(),
        "implementacion": platform.python_implementation(),
        "sistema": platform.system(),
        "plataforma": platform.platform(),
        "arquitectura": platform.machine(),
        "en_contenedor": Path("/.dockerenv").exists() or Path("/run/.containerenv").exists(),
        "paquetes": paquetes,
    }


def comparar_predicciones(esperado, obtenido, tolerancia: float = 1e-9) -> dict:
    """Compara dos vectores de predicciones fila por fila."""
    esperado = np.asarray(esperado, dtype=float)
    obtenido = np.asarray(obtenido, dtype=float)
    if esperado.shape != obtenido.shape:
        raise ValueError(f"Tamaños distintos: {esperado.shape} vs {obtenido.shape}")
    dif = np.abs(esperado - obtenido)
    return {
        "filas": int(len(dif)),
        "max_abs_diff": float(dif.max()) if len(dif) else 0.0,
        "media_abs_diff": float(dif.mean()) if len(dif) else 0.0,
        "filas_fuera_tolerancia": int((dif > tolerancia).sum()),
        "identicas_bit_a_bit": bool(np.array_equal(esperado, obtenido)),
        "tolerancia": float(tolerancia),
        "dentro_tolerancia": bool((dif <= tolerancia).all()),
    }


def _modelo_dir(modelo_dir: str | Path | None) -> Path:
    if modelo_dir:
        return ruta_proyecto(modelo_dir)
    return ruta_proyecto(os.getenv("MODELO_DIR") or cargar_params("exportar")["destino"])


def verificar_modelo(modelo_dir: str | Path | None = None, referencia: str | Path | None = None,
                     tolerancia: float | None = None) -> dict:
    """Predice el *golden set* con el modelo exportado y lo compara con lo esperado."""
    from tc5044_mlops.modeling.evaluate import metricas_regresion
    from tc5044_mlops.modeling.exportar import cargar_model_info, cargar_modelo_local

    modelo_dir = _modelo_dir(modelo_dir)
    tolerancia = cargar_params("exportar")["tolerancia"] if tolerancia is None else tolerancia
    info = cargar_model_info(modelo_dir)
    ref = pd.read_parquet(referencia or modelo_dir / info["referencia"]["archivo"])
    modelo = cargar_modelo_local(modelo_dir)

    inicio = time.perf_counter()
    pred = np.clip(modelo.predict(ref[info["columnas_entrada"]]), 0, None)
    segundos = time.perf_counter() - inicio

    resultado = {
        "verificacion": "prediccion_golden_set",
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo": {k: info.get(k) for k in ("nombre", "version", "model_uri", "run_id")},
        "comparacion": comparar_predicciones(ref["prediccion_esperada"], pred, tolerancia),
        "metricas_referencia_esperadas": info["referencia"].get("metricas"),
        "metricas_referencia_obtenidas": metricas_regresion(ref["demanda"], pred),
        "segundos_inferencia": segundos,
        "entorno_actual": huella_entorno(),
        "entorno_exportacion": info.get("entorno_exportacion"),
    }
    estado = "OK" if resultado["comparacion"]["dentro_tolerancia"] else "DIFERENTE"
    logger.info(f"Verificación {estado}: max_abs_diff="
                f"{resultado['comparacion']['max_abs_diff']:.3e} en {len(ref):,} filas")
    return resultado


def reentrenar_y_comparar(modelo_dir: str | Path | None = None, tabla: pd.DataFrame | None = None,
                          params: dict | None = None, tolerancia: float | None = None) -> dict:
    """Re-entrena un clon del `Pipeline` exportado y compara con el original.

    ``sklearn.base.clone`` copia los hiperparámetros (incluidas las semillas)
    sin los parámetros aprendidos; si el entrenamiento es determinista, el
    clon re-entrenado debe dar exactamente las mismas predicciones.
    """
    from sklearn.base import clone

    from tc5044_mlops.features import cargar_tabla
    from tc5044_mlops.modeling.evaluate import metricas_regresion
    from tc5044_mlops.modeling.exportar import cargar_model_info, cargar_modelo_local, particionar

    params = params or cargar_params()
    fijar_semillas(params["semilla"])
    modelo_dir = _modelo_dir(modelo_dir)
    tolerancia = params["exportar"]["tolerancia"] if tolerancia is None else tolerancia
    info = cargar_model_info(modelo_dir)
    original = cargar_modelo_local(modelo_dir)
    tabla = cargar_tabla() if tabla is None else tabla
    datos = particionar(tabla, params)

    inicio = time.perf_counter()
    clon = clone(original).fit(datos["X_tr"], datos["y_tr"])
    segundos = time.perf_counter() - inicio

    metricas = (metricas_regresion(datos["y_va"], clon.predict(datos["X_va"]), "val_")
                | metricas_regresion(datos["y_te"], clon.predict(datos["X_te"]), "test_"))
    esperadas = info.get("metricas_exportacion", {})
    dif_metricas = {k: abs(metricas[k] - esperadas[k]) for k in metricas if k in esperadas}

    ref = pd.read_parquet(modelo_dir / info["referencia"]["archivo"])
    pred_ref = np.clip(clon.predict(ref[info["columnas_entrada"]]), 0, None)
    resultado = {
        "verificacion": "reentrenamiento_con_semilla",
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo": {k: info.get(k) for k in ("nombre", "version", "model_uri", "run_id")},
        "semilla": params["semilla"],
        "filas_train": len(datos["X_tr"]),
        "segundos_entrenamiento": segundos,
        "metricas_reentrenado": metricas,
        "metricas_originales": esperadas,
        "max_dif_metricas": max(dif_metricas.values()) if dif_metricas else None,
        "comparacion_golden_set": comparar_predicciones(ref["prediccion_esperada"], pred_ref,
                                                        tolerancia),
        "entorno_actual": huella_entorno(),
        "entorno_exportacion": info.get("entorno_exportacion"),
    }
    logger.info(f"Re-entrenamiento: val_mae={metricas['val_mae']:.6f} "
                f"(original {esperadas.get('val_mae', float('nan')):.6f}), "
                f"max_dif_metricas={resultado['max_dif_metricas']}")
    return resultado


def guardar_resultado(resultado: dict, salida: str | Path | None, nombre: str) -> None:
    """Escribe el JSON; con ``salida='-'`` lo imprime en stdout (útil dentro del contenedor)."""
    from tc5044_mlops.tracking import a_json

    texto = a_json(resultado)
    if str(salida) == "-":
        sys.stdout.write(json.dumps(json.loads(texto), indent=2, ensure_ascii=False) + "\n")
        return
    destino = Path(salida) if salida else (
        REPORTS_DIR / "reproducibilidad" / f"{nombre}_{os.getenv('ENTORNO', 'local')}.json")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(json.loads(texto), indent=2, ensure_ascii=False),
                       encoding="utf-8")
    logger.success(f"Resultado -> {destino}")


def _logs_a_stderr() -> None:
    """Manda los logs a stderr para que ``--salida -`` deje en stdout solo el JSON."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")


@app.command("verificar")
def cli_verificar(
    modelo_dir: Path = typer.Option(None, help="Carpeta del modelo (por defecto MODELO_DIR "
                                               "o exportar.destino)"),
    salida: str = typer.Option(None, help="Archivo JSON de salida o '-' para stdout"),
) -> None:
    """Predice el golden set con el modelo exportado y compara con lo esperado."""
    _logs_a_stderr()
    resultado = verificar_modelo(modelo_dir)
    guardar_resultado(resultado, salida, "verificacion_modelo")
    if not resultado["comparacion"]["dentro_tolerancia"]:
        raise typer.Exit(code=1)


@app.command("reentrenar")
def cli_reentrenar(
    modelo_dir: Path = typer.Option(None, help="Carpeta del modelo exportado"),
    salida: str = typer.Option(None, help="Archivo JSON de salida o '-' para stdout"),
) -> None:
    """Re-entrena un clon del modelo con la misma semilla y compara métricas y predicciones."""
    _logs_a_stderr()
    guardar_resultado(reentrenar_y_comparar(modelo_dir), salida, "reentrenamiento")


@app.command("huella")
def cli_huella() -> None:
    """Imprime la huella del entorno (Python, sistema y versiones de paquetes)."""
    _logs_a_stderr()
    sys.stdout.write(json.dumps(huella_entorno(), indent=2) + "\n")


if __name__ == "__main__":
    app()
