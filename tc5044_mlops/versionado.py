"""
Versionado y trazabilidad de datasets con MLflow (Fase 1, refactorizado en la Fase 2).

Registra cada versión del dataset de MiBici como una corrida de MLflow con:
  - el dataset como entrada formal (mlflow.data.from_pandas + log_input),
    que deja asentados esquema, perfil y digest del conjunto;
  - el SHA-256 del archivo completo como etiqueta. El digest de MLflow solo
    mira las primeras 10 000 filas: en la Fase 1 no detectó una corrección
    que empezaba en la fila 14 592; el SHA-256 sí;
  - los parámetros que produjeron esa versión;
  - métricas de calidad comparables entre versiones;
  - el reporte de limpieza como artefacto.

Uso (desde la raíz del proyecto):
    python -m tc5044_mlops.versionado
    python -m tc5044_mlops.versionado --experimento mibici-datasets

Antes vivía en scripts/versionar_mlflow.py.
"""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import mlflow
import pandas as pd
import typer

from tc5044_mlops.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, tracking_uri
from tc5044_mlops.tracking import sha256_archivo

app = typer.Typer(help="Versionado de datasets en MLflow.")

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

EXPERIMENTO = "mibici-versionado-datasets"


def perfil_calidad(df: pd.DataFrame, etiqueta: str) -> dict[str, float]:
    """Métricas comparables entre versiones del dataset."""
    n = len(df)
    m: dict[str, float] = {
        "filas": n,
        "columnas": df.shape[1],
        "memoria_mb": round(df.memory_usage(deep=True).sum() / 1048576, 2),
        "viajes_id_unicos": int(df["viaje_id"].nunique()),
        "usuarios_unicos": int(df["usuario_id"].nunique()),
        "estaciones_unicas": int(
            pd.concat([df["origen_id"], df["destino_id"]]).nunique()
        ),
        "pct_nulos_total": round(100 * df.isna().sum().sum() / (n * df.shape[1]), 4),
    }
    for col in ("genero", "anio_nacimiento"):
        if col in df.columns:
            m[f"nulos_{col}"] = int(df[col].isna().sum())
    if "duracion_min" in df.columns:
        d = df["duracion_min"]
        m |= {
            "duracion_media": round(float(d.mean()), 3),
            "duracion_mediana": round(float(d.median()), 3),
            "duracion_p95": round(float(d.quantile(0.95)), 3),
            "duracion_max": round(float(d.max()), 3),
        }
    if "viaje_valido" in df.columns:
        m["viajes_validos"] = int(df["viaje_valido"].sum())
        m["pct_validos"] = round(100 * df["viaje_valido"].mean(), 4)
    print(f"  [{etiqueta}] {n:,} filas, {m['pct_nulos_total']} % nulos")
    return m


def registrar_version(
    ruta: Path,
    nombre: str,
    etapa: str,
    params: dict | None = None,
    artefactos: list[Path] | None = None,
) -> str | None:
    """Crea una corrida de MLflow que documenta una versión del dataset."""
    if not ruta.exists():
        print(f"  [omito] no existe {ruta}")
        return None

    df = pd.read_parquet(ruta)

    with mlflow.start_run(run_name=nombre) as run:
        mlflow.set_tags({
            "etapa": etapa,
            "fuente": "MiBici datos abiertos",
            "fuente_url": "https://www.mibici.net/es/datos-abiertos/",
            "periodo_inicio": str(df["periodo"].min()),
            "periodo_fin": str(df["periodo"].max()),
            "formato": "parquet/zstd",
            "dataset_sha256": sha256_archivo(ruta),
        })

        # El dataset como entrada formal: MLflow guarda esquema, perfil y digest.
        dataset = mlflow.data.from_pandas(
            df,
            source=str(ruta),
            name=nombre,
            digest=None,  # MLflow lo calcula sobre el contenido
        )
        mlflow.log_input(dataset, context=etapa)

        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(perfil_calidad(df, nombre))

        # Distribución por periodo: deja ver la cobertura temporal de la versión
        por_mes = df.groupby("periodo").size().to_dict()
        mlflow.log_dict({str(k): int(v) for k, v in por_mes.items()},
                        "cobertura_temporal.json")
        mlflow.log_text(
            df.dtypes.astype(str).to_string(), "esquema.txt"
        )

        for a in artefactos or []:
            if a.exists():
                mlflow.log_artifact(str(a))

        print(f"  -> run_id {run.info.run_id}")
        return run.info.run_id


@app.command()
def main(
    interim: Path = typer.Option(INTERIM_DATA_DIR / "mibici_viajes.parquet"),
    processed: Path = typer.Option(PROCESSED_DATA_DIR / "mibici_viajes.parquet"),
    reporte: Path = typer.Option(REPORTS_DIR / "limpieza.json"),
    experimento: str = typer.Option(EXPERIMENTO),
) -> None:
    """Registra las versiones v1 (consolidado) y v2 (limpio) del dataset."""
    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experimento)
    print(f"tracking: {uri}  |  experimento: {experimento}\n")

    rep = {}
    if (r := Path(reporte)).exists():
        rep = json.loads(r.read_text(encoding="utf-8"))

    runs = {}

    print("v1 — dataset consolidado (fiel a la fuente)")
    runs["v1_interim"] = registrar_version(
        Path(interim),
        nombre="mibici-v1-consolidado",
        etapa="consolidado",
        params={
            "periodos_descargados": rep.get("viajes_entrada", "n/d"),
            "transformacion": "union de CSV mensuales, esquema normalizado",
        },
    )

    print("\nv2 — dataset limpio (banderas de validez + features temporales)")
    runs["v2_processed"] = registrar_version(
        Path(processed),
        nombre="mibici-v2-limpio",
        etapa="limpio",
        params=rep.get("params", {}) | {
            "duplicados_eliminados": rep.get("duplicados_eliminados", "n/d"),
            "regla_validez": "1 <= duracion_min <= 1440",
        },
        artefactos=[Path(reporte)],
    )

    ui = uri if uri.startswith("http") else f"mlflow ui --backend-store-uri {uri}"
    print(f"\nListo. Experimento '{experimento}' con {len([r for r in runs.values() if r])} "
          f"versiones registradas.")
    print(f"UI: {ui}")
    print("En cada corrida, la pestaña 'Datasets' muestra esquema, perfil y digest.")
    print("Para comparar versiones: selecciona ambas corridas y pulsa 'Compare'.")
    print("\n>>> EVIDENCIA: captura esa pantalla de comparación para el entregable.")


if __name__ == "__main__":
    app()
