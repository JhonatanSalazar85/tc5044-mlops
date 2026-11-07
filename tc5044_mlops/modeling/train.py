"""Entrenamiento, ajuste, evaluación y registro de modelos en MLflow.

Flujo (una corrida padre por ejecución, con corridas hijas por modelo):

    fase2-entrenamiento-<fecha>            <- corrida padre: dataset, partición, resumen
    ├── baseline                           <- un hijo por modelo candidato
    ├── ridge
    ├── poisson
    ├── random_forest
    ├── hgb
    ├── optuna-hgb                         <- búsqueda de hiperparámetros
    │   ├── trial-00 ... trial-NN          <- un nieto por intento de Optuna
    └── hgb_optuna                         <- HGB con los mejores hiperparámetros

Cada hijo registra: hiperparámetros, métricas de validación y de prueba,
tiempo de entrenamiento, figuras y el `Pipeline` completo como modelo de
MLflow (con firma y ejemplo de entrada). Al final, los modelos se registran
en el Model Registry y el de menor ``val_mae`` recibe el alias ``champion``;
el segundo, ``challenger``. La selección usa **solo validación**; prueba se
reporta para estimar el desempeño fuera de muestra.

Uso:
    python -m tc5044_mlops.modeling.train                      # todo, con Optuna
    python -m tc5044_mlops.modeling.train --n-trials 0         # sin Optuna
    python -m tc5044_mlops.modeling.train --modelos ridge,hgb  # subconjunto
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import pandas as pd
import typer

from tc5044_mlops import plots
from tc5044_mlops.config import FIGURES_DIR, REPORTS_DIR, cargar_params, ruta_proyecto
from tc5044_mlops.features import cargar_tabla, particion_temporal
from tc5044_mlops.modeling.evaluate import metricas_regresion
from tc5044_mlops.modeling.models import (
    JUSTIFICACION,
    SIN_PREPROCESADOR,
    crear_modelo,
    nombres_modelos,
)
from tc5044_mlops.modeling.tune import optimizar_hgb
from tc5044_mlops.pipeline import agrupar_estaciones, columnas_entrada, construir_pipeline
from tc5044_mlops.tracking import (
    TIPOS_PROPIOS,
    a_json,
    asignar_alias,
    configurar_mlflow,
    describir_modelo_registrado,
    etiquetas_trazabilidad,
    hiperparametros_simples,
    registrar_version,
    requisitos_modelo,
    ruta_codigo_paquete,
)

app = typer.Typer(help="Entrenamiento y registro de modelos de demanda MiBici.")


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
def preparar_datos(tabla: pd.DataFrame, params: dict) -> dict:
    """Partición cronológica y separación X / y con las columnas del pipeline."""
    cols = columnas_entrada(params["pipeline"])
    objetivo = params["pipeline"]["objetivo"]
    tr, va, te = particion_temporal(tabla, **params["particion"])
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError("Algún conjunto de la partición quedó vacío; revisa params.yaml")
    return {
        "X_tr": tr[cols], "y_tr": tr[objetivo],
        "X_va": va[cols], "y_va": va[objetivo],
        "X_te": te[cols], "y_te": te[objetivo],
        "meta_te": te[["origen_id", "fecha_hora"]],
    }


# --------------------------------------------------------------------------
# Un modelo candidato = una corrida hija
# --------------------------------------------------------------------------
def entrenar_candidato(nombre: str, datos: dict, params: dict,
                       hiperparametros: dict | None = None,
                       algoritmo: str | None = None) -> dict:
    """Entrena, evalúa y registra en MLflow un modelo candidato.

    ``algoritmo`` es la entrada del catálogo (``hgb_optuna`` usa ``hgb``).
    """
    algoritmo = algoritmo or nombre
    semilla = params["semilla"]
    pipe = construir_pipeline(crear_modelo(algoritmo, hiperparametros, semilla),
                              cfg=params["pipeline"], semilla=semilla,
                              preprocesar=algoritmo not in SIN_PREPROCESADOR)

    with mlflow.start_run(run_name=nombre, nested=True) as run:
        mlflow.set_tags({"modelo": nombre, "algoritmo": algoritmo,
                         "justificacion": JUSTIFICACION[algoritmo],
                         "tipo_modelo": type(pipe.named_steps["modelo"]).__name__})
        mlflow.log_params(hiperparametros_simples(pipe.named_steps["modelo"]))
        if algoritmo not in SIN_PREPROCESADOR:
            cfg = params["pipeline"]
            mlflow.log_params({"pre__pca_componentes": cfg["pca_componentes"],
                               "pre__target_encoder_cv": cfg["target_encoder_cv"],
                               "pre__n_numericas": len(cfg["numericas"]),
                               "pre__escalado": "StandardScaler"})

        inicio = time.perf_counter()
        pipe.fit(datos["X_tr"], datos["y_tr"])
        segundos = time.perf_counter() - inicio

        pred_va = pipe.predict(datos["X_va"])
        pred_te = pipe.predict(datos["X_te"])
        metricas = (metricas_regresion(datos["y_va"], pred_va, "val_")
                    | metricas_regresion(datos["y_te"], pred_te, "test_")
                    | {"tiempo_entrenamiento_seg": segundos})
        mlflow.log_metrics(metricas)

        for fig, archivo in (
            (plots.fig_real_vs_predicho(datos["y_va"], pred_va, f"{nombre}: validación"),
             "real_vs_predicho_val.png"),
            (plots.fig_error_por_hora(datos["y_va"], pred_va, datos["X_va"]["hora"],
                                      f"{nombre}: MAE por hora (validación)"),
             "error_por_hora_val.png"),
        ):
            mlflow.log_figure(fig, f"figuras/{archivo}")
            plt.close(fig)

        ejemplo = datos["X_va"].head(5)
        firma = infer_signature(datos["X_va"].head(200), pipe.predict(datos["X_va"].head(200)))
        info = mlflow.sklearn.log_model(pipe, name="modelo", signature=firma,
                                        input_example=ejemplo,
                                        skops_trusted_types=TIPOS_PROPIOS,
                                        pip_requirements=requisitos_modelo(),
                                        code_paths=[ruta_codigo_paquete()])

    logger.info(f"{nombre:15s} val_mae={metricas['val_mae']:.4f}  "
                f"test_mae={metricas['test_mae']:.4f}  ({segundos:.1f} s)")
    return {"modelo": nombre, "algoritmo": algoritmo, "run_id": run.info.run_id,
            "model_uri": info.model_uri, "pipeline": pipe, "pred_te": pred_te,
            "hiperparametros": hiperparametros_simples(pipe.named_steps["modelo"], ""),
            **metricas}


def buscar_hiperparametros(datos: dict, params: dict, n_trials: int,
                           dir_figuras: Path = FIGURES_DIR) -> dict:
    """Corre Optuna dentro de una corrida hija; cada intento queda como nieto."""
    with mlflow.start_run(run_name="optuna-hgb", nested=True):
        mlflow.set_tags({"modelo": "optuna-hgb", "tipo": "busqueda_hiperparametros",
                         "muestreador": "TPE"})
        mlflow.log_params({"n_trials": n_trials,
                           "espacio": a_json(params["optuna"]["espacio"])})

        def registrar_intento(numero: int, hp: dict, met: dict) -> None:
            with mlflow.start_run(run_name=f"trial-{numero:02d}", nested=True):
                mlflow.set_tag("tipo", "intento_optuna")
                mlflow.log_params(hp)
                mlflow.log_metrics(met)

        estudio = optimizar_hgb(datos["X_tr"], datos["y_tr"], datos["X_va"], datos["y_va"],
                                n_trials=n_trials, cfg=params["optuna"],
                                semilla=params["semilla"],
                                al_terminar_intento=registrar_intento)
        mlflow.log_params({f"mejor__{k}": v for k, v in estudio.best_params.items()})
        mlflow.log_metric("mejor_val_mae", estudio.best_value)

        historial = pd.DataFrame({"intento": [t.number for t in estudio.trials],
                                  "val_mae": [t.value for t in estudio.trials]})
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(historial["intento"], historial["val_mae"], "o", alpha=0.6, label="intento")
        ax.plot(historial["intento"], historial["val_mae"].cummin(), color="black",
                label="mejor hasta el momento")
        ax.set(xlabel="intento de Optuna", ylabel="val_mae", title="Historial de Optuna (HGB)")
        ax.legend()
        fig.tight_layout()
        mlflow.log_figure(fig, "figuras/optuna_historial.png")
        fig.savefig(dir_figuras / "fase2_optuna_historial.png", dpi=120)
        plt.close(fig)
    return estudio.best_params


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------
def ejecutar(tabla: pd.DataFrame | None = None, ruta_tabla: str | Path | None = None,
             modelos: list[str] | None = None, n_trials: int | None = None,
             registrar: bool = True, params: dict | None = None,
             salida_reporte: Path | None = None,
             dir_figuras: Path | None = None) -> pd.DataFrame:
    """Entrena todos los candidatos, elige campeón y registra en el Model Registry.

    Devuelve la tabla resumen (una fila por modelo).
    """
    params = params or cargar_params()
    ruta_tabla = ruta_proyecto(ruta_tabla or params["features"]["salida"])
    tabla = cargar_tabla(ruta_tabla) if tabla is None else tabla
    modelos = modelos or nombres_modelos()
    n_trials = params["optuna"]["n_trials"] if n_trials is None else n_trials
    nombre_registro = params["mlflow"]["modelo_registrado"]
    metrica = params["mlflow"]["metrica_seleccion"]
    dir_figuras = Path(dir_figuras or FIGURES_DIR)
    dir_figuras.mkdir(parents=True, exist_ok=True)

    configurar_mlflow(params)
    datos = preparar_datos(tabla, params)

    with mlflow.start_run(run_name=f"fase2-entrenamiento-{datetime.now():%Y%m%d-%H%M}"):
        mlflow.set_tags(etiquetas_trazabilidad(ruta_tabla, tabla) | {"tipo": "corrida_padre"})
        mlflow.log_params({"semilla": params["semilla"], **{
            f"particion__{k}": v for k, v in params["particion"].items()},
            "top_estaciones": params["features"]["top_estaciones"],
            "filas_train": len(datos["X_tr"]), "filas_val": len(datos["X_va"]),
            "filas_test": len(datos["X_te"]),
            "estaciones": int(tabla["origen_id"].nunique()),
            # nombres distintos a los parámetros de MLproject (mlflow run ya los registra)
            "modelos_entrenados": ",".join(modelos), "n_trials_optuna": n_trials})
        mlflow.log_input(mlflow.data.from_pandas(
            tabla, source=str(ruta_tabla), name="mibici-demanda-horaria",
            targets=params["pipeline"]["objetivo"]), context="entrenamiento")
        mlflow.log_dict(params, "params.yaml")

        resultados = [entrenar_candidato(m, datos, params) for m in modelos]

        if n_trials > 0 and "hgb" in modelos:
            mejores = buscar_hiperparametros(datos, params, n_trials, dir_figuras)
            resultados.append(entrenar_candidato("hgb_optuna", datos, params,
                                                 hiperparametros=mejores, algoritmo="hgb"))

        resumen = (pd.DataFrame([{k: v for k, v in r.items()
                                  if k not in ("pipeline", "pred_te", "hiperparametros")}
                                 for r in resultados])
                   .sort_values(metrica).reset_index(drop=True))
        mlflow.log_table(resumen, "resumen_modelos.json")
        fig = plots.fig_comparacion_modelos(resumen, metrica)
        mlflow.log_figure(fig, "figuras/comparacion_modelos.png")
        fig.savefig(dir_figuras / "fase2_comparacion_modelos.png", dpi=120)
        plt.close(fig)

        por_nombre = {r["modelo"]: r for r in resultados}
        candidatos = [r for r in resumen["modelo"] if por_nombre[r]["algoritmo"]
                      not in SIN_PREPROCESADOR]
        campeon = por_nombre[candidatos[0]] if candidatos else None
        if campeon:
            _figuras_campeon(campeon, datos, params, dir_figuras)
            mlflow.set_tag("campeon", campeon["modelo"])

        versiones = {}
        if registrar and candidatos:
            describir_modelo_registrado(
                nombre_registro,
                "Demanda de retiros MiBici por estación y hora (horizonte 1 h). "
                "Pipeline scikit-learn: imputación + StandardScaler, TargetEncoder de la "
                "estación y PCA del perfil horario. Alias: @champion (menor val_mae) y "
                "@challenger (segundo lugar).")
            for nombre in candidatos:
                r = por_nombre[nombre]
                versiones[nombre] = registrar_version(
                    r["model_uri"], nombre_registro,
                    etiquetas={"modelo": nombre, "algoritmo": r["algoritmo"],
                               "run_id": r["run_id"],
                               "val_mae": round(r["val_mae"], 4),
                               "val_rmse": round(r["val_rmse"], 4),
                               "val_wape": round(r["val_wape"], 4),
                               "test_mae": round(r["test_mae"], 4),
                               "hiperparametros": a_json(r["hiperparametros"])},
                    descripcion=f"{nombre}: {JUSTIFICACION[r['algoritmo']]} "
                                f"val_mae={r['val_mae']:.4f}, test_mae={r['test_mae']:.4f}.")
            asignar_alias(nombre_registro, "champion", versiones[candidatos[0]])
            if len(candidatos) > 1:
                asignar_alias(nombre_registro, "challenger", versiones[candidatos[1]])
        resumen["version_registro"] = resumen["modelo"].map(versiones)

        reporte = {
            "metrica_seleccion": metrica,
            "campeon": campeon["modelo"] if campeon else None,
            "modelo_registrado": nombre_registro if versiones else None,
            "modelos": resumen.drop(columns=["model_uri"]).to_dict(orient="records"),
        }
        salida_reporte = salida_reporte or REPORTS_DIR / "metricas_fase2.json"
        salida_reporte.parent.mkdir(parents=True, exist_ok=True)
        salida_reporte.write_text(a_json(reporte).replace("NaN", "null"), encoding="utf-8")
        mlflow.log_artifact(str(salida_reporte))

    logger.success("Resumen (ordenado por {}):\n{}".format(
        metrica, resumen[["modelo", "val_mae", "val_rmse", "val_wape", "test_mae",
                          "tiempo_entrenamiento_seg"]].to_string(index=False)))
    return resumen


def _figuras_campeon(campeon: dict, datos: dict, params: dict, dir_figuras: Path) -> None:
    """Figuras del modelo campeón (prueba) y del análisis de perfiles PCA."""
    meta = datos["meta_te"].assign(real=datos["y_te"].to_numpy(), pred=campeon["pred_te"])
    estacion = meta.groupby("origen_id")["real"].sum().idxmax()
    sub = meta[meta["origen_id"] == estacion]
    figs = {
        "fase2_campeon_serie_test.png": plots.fig_serie_estacion(
            sub["fecha_hora"], sub["real"], sub["pred"], estacion),
        "fase2_campeon_real_vs_predicho_test.png": plots.fig_real_vs_predicho(
            meta["real"], meta["pred"], f"Campeón ({campeon['modelo']}): prueba"),
    }
    pre = campeon["pipeline"].named_steps.get("preprocesador")
    if pre is not None:
        pca = pre.named_transformers_["perfil_pca"].named_steps["pca"]
        grupos = agrupar_estaciones(pca, params["pipeline"]["clusters_estaciones"],
                                    params["semilla"])
        figs["fase2_perfiles_pca_estaciones.png"] = plots.fig_perfiles_pca(
            grupos, pca.tabla_perfiles(), pca.varianza_explicada_)
        mlflow.log_table(grupos.reset_index(), "grupos_estaciones.json")
        mlflow.log_metric("pca_varianza_explicada_total", float(pca.varianza_explicada_.sum()))
    for archivo, fig in figs.items():
        mlflow.log_figure(fig, f"figuras/{archivo}")
        fig.savefig(dir_figuras / archivo, dpi=120)
        plt.close(fig)


@app.command()
def main(
    modelos: str = typer.Option("todos", help="Lista separada por comas o 'todos'"),
    n_trials: int = typer.Option(None, help="Intentos de Optuna (0 = sin búsqueda)"),
    tabla: Path = typer.Option(None, help="Tabla de modelado (por defecto params.yaml)"),
    sin_registro: bool = typer.Option(False, help="No registrar en el Model Registry"),
) -> None:
    """Entrena los modelos candidatos, los compara en MLflow y registra el campeón."""
    lista = None if modelos == "todos" else [m.strip() for m in modelos.split(",") if m.strip()]
    resumen = ejecutar(ruta_tabla=tabla, modelos=lista, n_trials=n_trials,
                       registrar=not sin_registro)
    print(json.dumps(resumen[["modelo", "val_mae", "test_mae"]].to_dict(orient="records"),
                     indent=2))


if __name__ == "__main__":
    app()
