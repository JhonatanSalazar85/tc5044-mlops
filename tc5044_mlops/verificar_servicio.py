"""Verifica un servicio en ejecución (local o en contenedor) contra el golden set del modelo.

Es la prueba de portabilidad de la Fase 3: el golden set se generó en el entorno
de entrenamiento (WSL) con ``make exportar``; aquí se envía por HTTP al servicio
que corre en **otro entorno** (la imagen de contenedor, con otro sistema base y
otra instalación de Python) y se comparan, fila por fila, las predicciones y las
métricas.

Comprueba además que el servicio sirve **el mismo artefacto** (SHA-256 del
modelo y URI del Registry iguales a los de ``model_info.json`` local) y mide la
latencia de ``POST /predict`` con un registro y con lotes.

Uso:
    python -m tc5044_mlops.verificar_servicio --url http://127.0.0.1:8000
    python -m tc5044_mlops.verificar_servicio --url http://127.0.0.1:8000 --salida reports/x.json
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import httpx
from loguru import logger
import numpy as np
import pandas as pd
import typer

from tc5044_mlops.config import REPORTS_DIR, cargar_params, ruta_proyecto
from tc5044_mlops.modeling.evaluate import metricas_regresion
from tc5044_mlops.reproducibilidad import comparar_predicciones

app = typer.Typer(help="Verificación de un servicio en ejecución contra el golden set.")

CAMPOS_ENTRADA = ["origen_id", "fecha_hora", "demanda_lag_1h", "demanda_lag_2h",
                  "demanda_lag_24h", "demanda_lag_168h", "demanda_media_3h",
                  "demanda_media_24h", "demanda_media_168h"]


def a_registros(ref: pd.DataFrame) -> list[dict]:
    """Filas del golden set -> registros JSON de ``POST /predict`` (NaN -> null)."""
    regs = ref[CAMPOS_ENTRADA].copy()
    regs["fecha_hora"] = pd.to_datetime(regs["fecha_hora"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    regs["origen_id"] = regs["origen_id"].astype(int)
    return regs.astype(object).where(regs.notna(), None).to_dict(orient="records")


def _percentiles(ms: list[float]) -> dict:
    if not ms:
        return {}
    arr = np.asarray(ms)
    return {"n": len(arr), "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)), "max_ms": float(arr.max())}


def verificar_servicio(cliente: httpx.Client, modelo_dir: str | Path,
                       tolerancia: float | None = None, lote: int = 500,
                       peticiones_unitarias: int = 30) -> dict:
    """Compara las respuestas del servicio con el golden set de ``modelo_dir``.

    ``cliente`` es cualquier ``httpx.Client`` con ``base_url`` del servicio (en las
    pruebas, el ``TestClient`` de FastAPI).
    """
    modelo_dir = Path(modelo_dir)
    tolerancia = cargar_params("exportar")["tolerancia"] if tolerancia is None else tolerancia
    info_local = json.loads((modelo_dir / "model_info.json").read_text(encoding="utf-8"))
    ref = pd.read_parquet(modelo_dir / info_local["referencia"]["archivo"])

    salud = cliente.get("/health")
    salud.raise_for_status()
    info_remota = cliente.get("/model-info").json()

    registros = a_registros(ref)
    predicciones, tiempos_lote = [], []
    for i in range(0, len(registros), lote):
        inicio = time.perf_counter()
        r = cliente.post("/predict", json={"registros": registros[i:i + lote]})
        tiempos_lote.append(1000 * (time.perf_counter() - inicio))
        r.raise_for_status()
        predicciones.extend(p["prediccion"] for p in r.json()["predicciones"])

    tiempos_uno = []
    for registro in registros[:peticiones_unitarias]:
        inicio = time.perf_counter()
        cliente.post("/predict", json={"registros": [registro]}).raise_for_status()
        tiempos_uno.append(1000 * (time.perf_counter() - inicio))

    pred = np.asarray(predicciones, dtype=float)
    objetivo = info_local.get("objetivo", "demanda")
    resultado = {
        "verificacion": "servicio_http_vs_golden_set",
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": str(cliente.base_url),
        "mismo_artefacto": {
            "model_uri_local": info_local["model_uri"],
            "model_uri_servicio": info_remota.get("model_uri"),
            "sha256_local": info_local.get("sha256_modelo"),
            "sha256_servicio": info_remota.get("sha256_modelo"),
            "coincide": (info_local.get("sha256_modelo") == info_remota.get("sha256_modelo")
                         and info_local["model_uri"] == info_remota.get("model_uri")),
        },
        "comparacion": comparar_predicciones(ref["prediccion_esperada"], pred, tolerancia),
        "metricas_esperadas": info_local["referencia"].get("metricas"),
        "metricas_servicio": metricas_regresion(ref[objetivo], pred),
        "latencia_lote": _percentiles(tiempos_lote) | {"registros_por_lote": lote},
        "latencia_un_registro": _percentiles(tiempos_uno),
        "salud": salud.json(),
    }
    ok = resultado["comparacion"]["dentro_tolerancia"] and resultado["mismo_artefacto"]["coincide"]
    logger.info(f"Servicio {'OK' if ok else 'DIFERENTE'}: "
                f"max_abs_diff={resultado['comparacion']['max_abs_diff']:.3e}, "
                f"p50 un registro={resultado['latencia_un_registro'].get('p50_ms', 0):.1f} ms")
    return resultado


@app.command()
def main(
    url: str = typer.Option("http://127.0.0.1:8000", help="URL base del servicio"),
    modelo_dir: Path = typer.Option(None, help="Carpeta local del modelo (golden set)"),
    salida: Path = typer.Option(None, help="JSON de salida"),
    nombre: str = typer.Option("contenedor", help="Etiqueta del entorno verificado"),
) -> None:
    """Envía el golden set al servicio y compara predicciones, métricas y artefacto."""
    modelo_dir = ruta_proyecto(modelo_dir or cargar_params("exportar")["destino"])
    with httpx.Client(base_url=url, timeout=60) as cliente:
        resultado = verificar_servicio(cliente, modelo_dir)
    resultado["entorno_verificado"] = nombre
    salida = salida or REPORTS_DIR / "reproducibilidad" / f"verificacion_servicio_{nombre}.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Resultado -> {salida}")
    ok = resultado["comparacion"]["dentro_tolerancia"] and resultado["mismo_artefacto"]["coincide"]
    if not ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
