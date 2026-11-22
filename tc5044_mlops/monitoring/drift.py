"""Simulación de data drift y detección de pérdida de desempeño del modelo campeón.

Flujo (``python -m tc5044_mlops.monitoring.drift`` o ``make drift``):

1. Carga el modelo exportado (``models/champion``) y la tabla de modelado.
2. Genera los conjuntos de monitoreo (``escenarios.generar_escenarios``):
   referencia = validación; control = prueba real; 4 escenarios alterados.
3. Para cada conjunto calcula:
   - drift de entradas por columna (PSI, KS, nulos) y fracción de columnas en alerta;
   - drift de la predicción (PSI de las salidas: se puede vigilar sin etiquetas);
   - inconsistencia de rezagos (regla de dominio);
   - desempeño (MAE, RMSE, WAPE, R²) y su cambio frente a la línea base.
4. Aplica los umbrales de ``params.yaml`` (sección ``drift``) y decide un nivel
   (OK / AVISO / ALERTA) y una **acción** (ninguna, vigilar, reentrenar o
   revisar el feature pipeline).
5. Curva de severidad: el desplazamiento de medias con factores crecientes, para
   ver qué detector se dispara primero.
6. Guarda JSON/CSV en ``reports/drift``, figuras ``reports/figures/fase3_drift_*``,
   reportes HTML de Evidently, y registra todo en MLflow (experimento
   ``mibici-monitoreo-fase3``).

En producción las etiquetas (los retiros reales) llegan con una hora de retraso,
así que el monitoreo de desempeño es viable casi en tiempo real; el drift de
entradas y de predicciones sirve como alerta temprana entre tanto.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from tc5044_mlops.config import FIGURES_DIR, cargar_params, ruta_proyecto
from tc5044_mlops.modeling.evaluate import metricas_regresion
from tc5044_mlops.monitoring import estadisticos as est
from tc5044_mlops.monitoring.escenarios import DESCRIPCION, desplazar_medias, generar_escenarios

app = typer.Typer(help="Simulación de data drift y alertas de desempeño.")

NIVELES = ["OK", "AVISO", "ALERTA"]
ACCIONES = {
    "ninguna": "Ninguna: el modelo opera dentro de los umbrales.",
    "vigilar": "Vigilar: repetir la evaluación con la siguiente ventana; no se reentrena aún.",
    "monitoreo_reforzado": "Drift en las entradas sin pérdida de desempeño: monitoreo reforzado "
                           "(ventanas diarias); reentrenar si persiste dos ventanas seguidas.",
    "revisar_pipeline": "Revisar el feature pipeline (ingesta y cálculo de rezagos) antes de "
                        "reentrenar: los datos de entrada están corruptos o incompletos. "
                        "Mitigación inmediata: servir la línea base estacional o un modelo sin "
                        "rezagos de 1-2 h.",
    "reentrenar": "Cambio real en la demanda: reentrenar con la ventana más reciente "
                  "(make train), comparar contra @champion en validación y promover solo si "
                  "mejora.",
    "reentrenar_e_investigar": "Pérdida de desempeño sin drift visible en las entradas (concept "
                               "drift): reentrenar e investigar variables externas (clima, "
                               "eventos, cierres de estaciones).",
}


# --------------------------------------------------------------------------
# Evaluación de un conjunto de monitoreo
# --------------------------------------------------------------------------
def decidir(r: dict, cfg: dict) -> tuple[str, str, list[str]]:
    """Nivel, acción y motivos a partir de los indicadores de un escenario."""
    motivos = []
    alerta_nulos = r["max_delta_nulos"] >= cfg["nulos_alerta"]
    alerta_incons = (np.isfinite(r["inconsistencia_rezagos"])
                     and r["inconsistencia_rezagos"] >= cfg["inconsistencia_alerta"])
    alerta_datos = r["fraccion_columnas_alerta"] >= cfg["fraccion_columnas_alerta"]
    cambio = r["cambio_mae"]
    alerta_mae, aviso_mae = cambio >= cfg["mae_alerta"], cambio >= cfg["mae_aviso"]
    if alerta_nulos:
        motivos.append(f"nulos +{100 * r['max_delta_nulos']:.0f} pp")
    if alerta_incons:
        motivos.append(f"rezagos inconsistentes en {100 * r['inconsistencia_rezagos']:.0f} % "
                       "de las filas")
    if r["n_columnas_alerta"]:
        motivos.append(f"{r['n_columnas_alerta']} de {r['n_columnas']} columnas en alerta "
                       "(PSI o nulos)")
    if r["n_columnas_aviso"]:
        motivos.append(f"{r['n_columnas_aviso']} columnas en aviso")
    if aviso_mae:
        motivos.append(f"MAE {100 * cambio:+.1f} % vs línea base")

    if alerta_nulos or alerta_incons:
        return "ALERTA", "revisar_pipeline", motivos
    if alerta_mae:
        # Si las entradas o el objetivo cambiaron, el mundo cambió: reentrenar. Si nada se
        # movió y aun así el error sube, la relación X -> y cambió (concept drift).
        cambio_visible = (alerta_datos or r["n_columnas_alerta"] > 0
                          or r["psi_objetivo"] >= cfg["psi_alerta"])
        return "ALERTA", ("reentrenar" if cambio_visible else "reentrenar_e_investigar"), motivos
    if alerta_datos:
        return "AVISO", "monitoreo_reforzado", motivos
    if aviso_mae or r["n_columnas_aviso"] > 0 or r["n_columnas_alerta"] > 0:
        return "AVISO", "vigilar", motivos
    return "OK", "ninguna", motivos


def evaluar_conjunto(nombre: str, referencia: pd.DataFrame, actual: pd.DataFrame, modelo,
                     columnas_entrada: list[str], cfg: dict, base: dict,
                     pred_ref: np.ndarray) -> dict:
    """Indicadores de drift y desempeño de un conjunto frente a la referencia."""
    pred = np.clip(modelo.predict(actual[columnas_entrada]), 0, None)
    columnas = est.resumen_columnas(referencia, actual, cfg["columnas"], cfg)
    m = metricas_regresion(actual["demanda"], pred)
    r = {
        "escenario": nombre,
        "descripcion": DESCRIPCION.get(nombre, ""),
        "filas": len(actual),
        "psi_max": float(np.nanmax(columnas["psi"])) if columnas["psi"].notna().any()
        else float("nan"),
        "psi_medio": float(np.nanmean(columnas["psi"])) if columnas["psi"].notna().any()
        else float("nan"),
        "ks_max": float(np.nanmax(columnas["ks"])) if columnas["ks"].notna().any()
        else float("nan"),
        "n_columnas": len(columnas),
        "n_columnas_alerta": int((columnas["estado"] == "alerta").sum()),
        "fraccion_columnas_alerta": float((columnas["estado"] == "alerta").mean()),
        "n_columnas_aviso": int((columnas["estado"] == "aviso").sum()),
        "max_delta_nulos": float(columnas["delta_nulos"].max()),
        "inconsistencia_rezagos": est.inconsistencia_rezagos(actual),
        "psi_prediccion": est.psi(pred_ref, pred, cfg["bins_psi"]),
        "psi_objetivo": est.psi(referencia["demanda"], actual["demanda"], cfg["bins_psi"]),
        "demanda_media": float(actual["demanda"].mean()),
        "prediccion_media": float(pred.mean()),
        **{k: float(v) for k, v in m.items()},
        "cambio_mae": float(m["mae"] / base["mae"] - 1),
        "cambio_wape": float(m["wape"] / base["wape"] - 1),
    }
    r["nivel"], r["accion"], r["motivos"] = decidir(r, cfg)
    r["accion_detalle"] = ACCIONES[r["accion"]]
    return {"resumen": r, "columnas": columnas, "prediccion": pred}


def curva_severidad(referencia: pd.DataFrame, modelo, columnas_entrada: list[str], cfg: dict,
                    base: dict, pred_ref: np.ndarray) -> pd.DataFrame:
    """Desplazamiento de medias con factores crecientes: PSI y MAE frente al factor."""
    esc = cfg["escenarios"]
    filas = []
    for factor in esc["severidad"]["factores"]:
        actual = desplazar_medias(referencia, esc["desplazamiento_medias"]["columnas"], factor)
        r = evaluar_conjunto(f"factor_{factor}", referencia, actual, modelo, columnas_entrada,
                             cfg, base, pred_ref)["resumen"]
        filas.append({"factor": factor, **{k: r[k] for k in (
            "psi_max", "psi_medio", "psi_prediccion", "fraccion_columnas_alerta", "mae",
            "cambio_mae", "inconsistencia_rezagos", "nivel", "accion")}})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# Evidently (reporte HTML para inspección visual)
# --------------------------------------------------------------------------
def reporte_evidently(referencia: pd.DataFrame, actual: pd.DataFrame, columnas: list[str],
                      destino: Path, n: int, semilla: int) -> dict | None:
    """Reporte ``DataDriftPreset`` de Evidently; devuelve la fracción de columnas con drift."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        logger.warning("Evidently no está instalado; se omite el reporte HTML")
        return None
    # Evidently no acepta columnas totalmente vacías: se reportan aparte (ya las detecta
    # la tasa de nulos) y el reporte se hace con las demás.
    vacias = [c for c in columnas if actual[c].isna().all()]
    usar = [c for c in columnas if c not in vacias]
    ref = referencia[usar].sample(min(n, len(referencia)), random_state=semilla)
    act = actual[usar].sample(min(n, len(actual)), random_state=semilla)
    try:
        snapshot = Report([DataDriftPreset(columns=usar)]).run(reference_data=ref,
                                                               current_data=act)
    except Exception as exc:  # noqa: BLE001 - el reporte HTML es complementario
        logger.warning(f"Evidently falló para {destino.name}: {exc}")
        return None
    destino.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(destino))
    salida = {"html": destino.name, "columnas": {}, "columnas_vacias": vacias}
    for metrica in snapshot.dict()["metrics"]:
        nombre = metrica.get("metric_name", "")
        if nombre.startswith("DriftedColumnsCount"):
            salida["columnas_con_drift"] = metrica["value"]["count"]
            salida["fraccion_con_drift"] = metrica["value"]["share"]
        elif nombre.startswith("ValueDrift"):
            columna = nombre.split("column=")[1].split(",")[0]
            salida["columnas"][columna] = float(metrica["value"])
    return salida


# --------------------------------------------------------------------------
# Figuras (paleta de referencia de la guía de visualización del equipo)
# --------------------------------------------------------------------------
TINTA, TINTA_2, TENUE, REJILLA = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIE_1, SERIE_2 = "#2a78d6", "#eb6834"
ESTADO = {"OK": "#0ca30c", "AVISO": "#fab219", "ALERTA": "#d03b3b"}
SECUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _estilo(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors=TINTA_2, labelsize=8)
    ax.grid(axis="y", color=REJILLA, linewidth=0.6)
    ax.set_axisbelow(True)


def figuras(resultados: dict, severidad: pd.DataFrame, conjuntos: dict, cfg: dict,
            base: dict, dir_figuras: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt

    dir_figuras.mkdir(parents=True, exist_ok=True)
    rutas = {}
    nombres = [n for n in resultados if n != "referencia"]

    # 1) PSI por columna y escenario (mapa de calor, rampa secuencial de un solo tono)
    columnas = cfg["columnas"] + ["prediccion"]
    matriz = np.array([[*resultados[n]["columnas"].set_index("columna")["psi"]
                        .reindex(cfg["columnas"]).to_numpy(),
                        resultados[n]["resumen"]["psi_prediccion"]] for n in nombres])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    cmap = LinearSegmentedColormap.from_list("azul", SECUENCIAL)
    vmax = 1.0      # escala acotada: >= 1 ya es un cambio extremo; así se distingue 0.1-0.5
    im = ax.imshow(np.clip(np.nan_to_num(matriz, nan=0.0), 0, vmax), cmap=cmap, vmin=0,
                   vmax=vmax, aspect="auto")
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            v = matriz[i, j]
            texto = "nulo" if not np.isfinite(v) else f"{v:.2f}"
            claro = np.isfinite(v) and v > 0.55 * vmax
            ax.text(j, i, texto, ha="center", va="center", fontsize=7,
                    color="#ffffff" if claro else TINTA)
    ax.set_xticks(range(len(columnas)), [c.replace("demanda_", "") for c in columnas],
                  rotation=30, ha="right", fontsize=8, color=TINTA_2)
    ax.set_yticks(range(len(nombres)), nombres, fontsize=8, color=TINTA_2)
    ax.set_title(f"PSI frente a validación (alerta ≥ {cfg['psi_alerta']}; 'nulo' = columna "
                 "sin valores)", fontsize=9, color=TINTA, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    barra = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, extend="max")
    barra.ax.tick_params(labelsize=7)
    barra.set_label("PSI (≥ 1 saturado)", fontsize=7, color=TINTA_2)
    fig.tight_layout()
    rutas["psi"] = dir_figuras / "fase3_drift_psi.png"
    fig.savefig(rutas["psi"], dpi=150)
    plt.close(fig)

    # 2) MAE por escenario frente a la línea base, coloreado por nivel (con etiqueta)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    maes = [resultados[n]["resumen"]["mae"] for n in nombres]
    niveles = [resultados[n]["resumen"]["nivel"] for n in nombres]
    ax.bar(range(len(nombres)), maes, width=0.6, color=[ESTADO[nv] for nv in niveles],
           edgecolor="#fcfcfb", linewidth=2)
    lineas = ((0, "-", "línea base (validación)"),
              (cfg["mae_aviso"], ":", f"aviso (+{100 * cfg['mae_aviso']:.0f} %)"),
              (cfg["mae_alerta"], "--", f"alerta (+{100 * cfg['mae_alerta']:.0f} %)"))
    for frac, estilo, etiqueta in lineas:
        ax.axhline(base["mae"] * (1 + frac), color=TINTA_2 if frac == 0 else TENUE,
                   linestyle=estilo, linewidth=1, label=etiqueta)
    cambios = [100 * resultados[n]["resumen"]["cambio_mae"] for n in nombres]
    etiquetas = [f"{n.replace('_', ' ')}\n{nv} · {c:+.0f} %"
                 for n, nv, c in zip(nombres, niveles, cambios)]
    ax.set_xticks(range(len(nombres)), etiquetas, fontsize=7.5)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_ylabel("MAE (bicis/h)", fontsize=8, color=TINTA_2)
    ax.set_ylim(0, max(max(maes), base["mae"] * (1 + cfg["mae_alerta"])) * 1.3)
    ax.set_title(f"Desempeño por escenario (línea base = validación, MAE {base['mae']:.3f})",
                 fontsize=9, color=TINTA, loc="left")
    _estilo(ax)
    fig.tight_layout()
    rutas["desempeno"] = dir_figuras / "fase3_drift_desempeno.png"
    fig.savefig(rutas["desempeno"], dpi=150)
    plt.close(fig)

    # 3) Severidad: dos paneles (nunca dos ejes y en el mismo gráfico)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    a1.plot(severidad["factor"], severidad["psi_max"], color=SERIE_1, linewidth=2, marker="o",
            markersize=5)
    a1.axhline(cfg["psi_alerta"], color=TENUE, linestyle="--", linewidth=1)
    a1.text(severidad["factor"].iloc[0], cfg["psi_alerta"], "  alerta PSI", va="bottom",
            fontsize=7, color=TINTA_2)
    a1.set_title("Drift detectado: PSI máximo de las entradas", fontsize=9, color=TINTA,
                 loc="left")
    a2.plot(severidad["factor"], 100 * severidad["cambio_mae"], color=SERIE_1, linewidth=2,
            marker="o", markersize=5)
    a2.axhline(100 * cfg["mae_alerta"], color=TENUE, linestyle="--", linewidth=1)
    a2.text(severidad["factor"].iloc[0], 100 * cfg["mae_alerta"], "  alerta MAE", va="bottom",
            fontsize=7, color=TINTA_2)
    a2.set_title("Impacto: cambio del MAE (%)", fontsize=9, color=TINTA, loc="left")
    for a in (a1, a2):
        a.set_xlabel("factor aplicado a rezagos y medias", fontsize=8, color=TINTA_2)
        _estilo(a)
    fig.tight_layout()
    rutas["severidad"] = dir_figuras / "fase3_drift_severidad.png"
    fig.savefig(rutas["severidad"], dpi=150)
    plt.close(fig)

    # 4) Distribución de la predicción: referencia vs cada escenario (múltiplos pequeños)
    fig, ejes = plt.subplots(1, len(nombres), figsize=(2.1 * len(nombres), 2.6), sharey=True)
    p_ref = resultados["referencia"]["prediccion"]
    lim = float(np.quantile(p_ref, 0.995)) * 1.2
    cortes = np.linspace(0, lim, 40)
    for ax, n in zip(np.atleast_1d(ejes), nombres):
        ax.hist(p_ref, cortes, density=True, histtype="step", color=SERIE_1, linewidth=1.5,
                label="validación")
        ax.hist(resultados[n]["prediccion"], cortes, density=True, histtype="step",
                color=SERIE_2, linewidth=1.5, label="escenario")
        ax.set_title(n.replace("_", "\n"), fontsize=7, color=TINTA)
        ax.set_yticks([])
        _estilo(ax)
    np.atleast_1d(ejes)[0].legend(fontsize=6, frameon=False, loc="upper right")
    fig.suptitle("Distribución de las predicciones (bicis/h): drift de salida sin etiquetas",
                 fontsize=9, color=TINTA, x=0.01, ha="left")
    fig.tight_layout()
    rutas["predicciones"] = dir_figuras / "fase3_drift_predicciones.png"
    fig.savefig(rutas["predicciones"], dpi=150)
    plt.close(fig)
    return rutas


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------
def ejecutar(tabla: pd.DataFrame | None = None, modelo=None, params: dict | None = None,
             modelo_dir: str | Path | None = None, salida: str | Path | None = None,
             dir_figuras: str | Path | None = None, con_evidently: bool = True,
             con_mlflow: bool = True) -> dict:
    """Corre todos los escenarios y devuelve el resumen (también lo escribe en disco)."""
    from tc5044_mlops.features import cargar_tabla
    from tc5044_mlops.modeling.exportar import cargar_model_info, cargar_modelo_local
    from tc5044_mlops.reproducibilidad import fijar_semillas

    params = params or cargar_params()
    cfg = params["drift"]
    fijar_semillas(params["semilla"])
    salida = ruta_proyecto(salida or cfg["salida"])
    dir_figuras = Path(dir_figuras) if dir_figuras else FIGURES_DIR
    salida.mkdir(parents=True, exist_ok=True)

    info = {}
    if modelo is None:
        modelo_dir = ruta_proyecto(modelo_dir or params["exportar"]["destino"])
        modelo, info = cargar_modelo_local(modelo_dir), cargar_model_info(modelo_dir)
    from tc5044_mlops.pipeline import columnas_entrada

    cols = info.get("columnas_entrada") or columnas_entrada(params["pipeline"])
    tabla = cargar_tabla() if tabla is None else tabla
    conjuntos = generar_escenarios(tabla, params)
    ref = conjuntos["referencia"]
    pred_ref = np.clip(modelo.predict(ref[cols]), 0, None)
    base = metricas_regresion(ref["demanda"], pred_ref)

    resultados = {n: evaluar_conjunto(n, ref, df, modelo, cols, cfg, base, pred_ref)
                  for n, df in conjuntos.items()}
    severidad = curva_severidad(ref, modelo, cols, cfg, base, pred_ref)

    evidently = {}
    if con_evidently:
        for n, df in conjuntos.items():
            if n != "referencia":
                evidently[n] = reporte_evidently(ref, df, cfg["columnas"],
                                                 salida / f"evidently_{n}.html",
                                                 cfg["muestra_evidently"], params["semilla"])

    for n, r in resultados.items():
        r["columnas"].to_csv(salida / f"columnas_{n}.csv", index=False)
        if evidently.get(n):
            r["resumen"]["evidently_fraccion_drift"] = evidently[n].get("fraccion_con_drift")
    severidad.to_csv(salida / "severidad.csv", index=False)
    rutas = figuras(resultados, severidad, conjuntos, cfg, base, dir_figuras)

    resumen = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "modelo": {k: info.get(k) for k in ("nombre", "version", "model_uri", "sha256_modelo")},
        "referencia": {"conjunto": "validación", "filas": len(ref),
                       "desde": str(ref["fecha_hora"].min()),
                       "hasta": str(ref["fecha_hora"].max()), "metricas": base},
        "umbrales": {k: cfg[k] for k in ("psi_aviso", "psi_alerta", "ks_alerta",
                                         "fraccion_columnas_alerta", "nulos_alerta",
                                         "inconsistencia_alerta", "mae_aviso", "mae_alerta")},
        "escenarios": [resultados[n]["resumen"] for n in resultados if n != "referencia"],
        "severidad": severidad.to_dict(orient="records"),
        "evidently": evidently,
        "figuras": {k: str(v) for k, v in rutas.items()},
    }
    from tc5044_mlops.tracking import a_json

    ruta_json = salida / "resumen_drift.json"
    ruta_json.write_text(json.dumps(json.loads(a_json(resumen).replace("NaN", "null")),
                                    indent=2, ensure_ascii=False), encoding="utf-8")
    (salida / "alertas.md").write_text(tabla_alertas(resumen), encoding="utf-8")
    # Métricas planas por escenario para `dvc metrics show` / `dvc metrics diff`
    planas = {r["escenario"]: {k: r[k] for k in ("mae", "wape", "cambio_mae", "psi_max",
                                                  "psi_prediccion", "fraccion_columnas_alerta",
                                                  "max_delta_nulos")}
              for r in resumen["escenarios"]}
    (salida / "metricas_drift.json").write_text(
        json.dumps(json.loads(a_json(planas).replace("NaN", "null")), indent=2), encoding="utf-8")
    if con_mlflow:
        registrar_en_mlflow(resumen, resultados, salida, rutas, params)
    logger.success(f"Drift: {ruta_json}\n" + tabla_alertas(resumen))
    return resumen


def tabla_alertas(resumen: dict) -> str:
    """Tabla Markdown escenario -> nivel, indicadores y acción."""
    lineas = ["| Escenario | PSI máx. | Nulos Δ | Incons. rezagos | MAE | Cambio MAE | Nivel | "
              "Acción |", "|---|---|---|---|---|---|---|---|"]
    for r in resumen["escenarios"]:
        inc = r["inconsistencia_rezagos"]
        psi_max = r["psi_max"]
        lineas.append(
            f"| {r['escenario']} | {psi_max:.3f} | {100 * r['max_delta_nulos']:.0f} pp | "
            f"{'—' if inc is None or not np.isfinite(inc) else f'{100 * inc:.0f} %'} | "
            f"{r['mae']:.3f} | {100 * r['cambio_mae']:+.1f} % | {r['nivel']} | {r['accion']} |")
    return "\n".join(lineas) + "\n"


def registrar_en_mlflow(resumen: dict, resultados: dict, salida: Path, rutas: dict,
                        params: dict) -> None:
    import mlflow

    from tc5044_mlops.tracking import configurar_mlflow

    p = dict(params)
    p["mlflow"] = dict(params["mlflow"], experimento=params["drift"]["experimento"])
    configurar_mlflow(p)
    with mlflow.start_run(run_name=f"fase3-drift-{datetime.now():%Y%m%d-%H%M}"):
        mlflow.set_tags({"tipo": "monitoreo_drift", "fase": "3",
                         "model_uri": str(resumen["modelo"].get("model_uri"))})
        mlflow.log_params({f"umbral__{k}": v for k, v in resumen["umbrales"].items()})
        mlflow.log_metrics({f"base_{k}": v for k, v in resumen["referencia"]["metricas"].items()})
        for r in resumen["escenarios"]:
            with mlflow.start_run(run_name=r["escenario"], nested=True):
                mlflow.set_tags({"escenario": r["escenario"], "nivel": r["nivel"],
                                 "accion": r["accion"], "descripcion": r["descripcion"]})
                mlflow.log_metrics({k: float(r[k]) for k in (
                    "mae", "rmse", "wape", "r2", "cambio_mae", "psi_max", "psi_medio",
                    "psi_prediccion", "fraccion_columnas_alerta", "max_delta_nulos")
                    if r.get(k) is not None and np.isfinite(r[k])})
                if r["inconsistencia_rezagos"] is not None and np.isfinite(
                        r["inconsistencia_rezagos"]):
                    mlflow.log_metric("inconsistencia_rezagos", r["inconsistencia_rezagos"])
                html = salida / f"evidently_{r['escenario']}.html"
                if html.exists():
                    mlflow.log_artifact(str(html), "evidently")
                mlflow.log_artifact(str(salida / f"columnas_{r['escenario']}.csv"))
        for ruta in rutas.values():
            mlflow.log_artifact(str(ruta), "figuras")
        for archivo in ("resumen_drift.json", "severidad.csv", "alertas.md"):
            mlflow.log_artifact(str(salida / archivo))


@app.command()
def main(
    modelo_dir: Path = typer.Option(None, help="Carpeta del modelo exportado"),
    sin_evidently: bool = typer.Option(False, help="No generar los reportes HTML de Evidently"),
    sin_mlflow: bool = typer.Option(False, help="No registrar la corrida en MLflow"),
) -> None:
    """Simula los escenarios de drift, evalúa el desempeño y emite alertas."""
    ejecutar(modelo_dir=modelo_dir, con_evidently=not sin_evidently, con_mlflow=not sin_mlflow)


if __name__ == "__main__":
    app()
