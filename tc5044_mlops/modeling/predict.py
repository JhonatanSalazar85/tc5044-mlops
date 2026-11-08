"""Inferencia con un modelo del MLflow Model Registry.

Carga el modelo por **alias** (``champion`` por defecto) en lugar de por
número de versión: cuando el entrenamiento promueve un modelo nuevo, este
script lo usa sin cambiar una línea de código.

Uso:
    python -m tc5044_mlops.modeling.predict                    # periodo de prueba
    python -m tc5044_mlops.modeling.predict --alias challenger
    python -m tc5044_mlops.modeling.predict --desde 2026-08-01 --salida reports/pred.parquet
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import mlflow
import pandas as pd
import typer

from tc5044_mlops.config import REPORTS_DIR, cargar_params
from tc5044_mlops.features import cargar_tabla
from tc5044_mlops.modeling.evaluate import metricas_regresion
from tc5044_mlops.pipeline import columnas_entrada
from tc5044_mlops.tracking import configurar_mlflow

app = typer.Typer(help="Predicciones con el modelo registrado en MLflow.")


def cargar_modelo(alias: str = "champion", params: dict | None = None):
    """Devuelve el `Pipeline` de scikit-learn registrado con el alias dado."""
    params = params or cargar_params()
    configurar_mlflow(params)
    uri = f"models:/{params['mlflow']['modelo_registrado']}@{alias}"
    logger.info(f"Cargando {uri}")
    return mlflow.sklearn.load_model(uri)


def predecir(tabla: pd.DataFrame, modelo, params: dict | None = None) -> pd.DataFrame:
    """Agrega la columna ``prediccion`` (bicis/h, nunca negativa) a la tabla."""
    params = params or cargar_params()
    X = tabla[columnas_entrada(params["pipeline"])]
    salida = tabla[["origen_id", "fecha_hora"]].copy()
    if params["pipeline"]["objetivo"] in tabla:
        salida["demanda"] = tabla[params["pipeline"]["objetivo"]].to_numpy()
    salida["prediccion"] = modelo.predict(X).clip(min=0)
    return salida


@app.command()
def main(
    alias: str = typer.Option("champion", help="Alias del Model Registry"),
    tabla: Path = typer.Option(None, help="Tabla de modelado (por defecto params.yaml)"),
    desde: str = typer.Option(None, help="Fecha inicial (por defecto el corte de prueba)"),
    salida: Path = typer.Option(None, help="Parquet de salida"),
) -> None:
    """Predice la demanda horaria con el modelo del alias indicado."""
    params = cargar_params()
    datos = cargar_tabla(tabla)
    desde = desde or params["particion"]["corte_test"]
    datos = datos[datos["fecha_hora"] >= desde]
    resultado = predecir(datos, cargar_modelo(alias, params), params)

    salida = salida or REPORTS_DIR / f"predicciones_{alias}.parquet"
    salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_parquet(salida, index=False)
    logger.success(f"{len(resultado):,} predicciones -> {salida}")
    if "demanda" in resultado:
        m = metricas_regresion(resultado["demanda"], resultado["prediccion"])
        logger.info("Métricas sobre el periodo predicho: " +
                    ", ".join(f"{k}={v:.4f}" for k, v in m.items()))


if __name__ == "__main__":
    app()
