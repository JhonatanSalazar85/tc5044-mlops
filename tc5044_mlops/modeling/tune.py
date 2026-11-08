"""Ajuste de hiperparámetros del HistGradientBoosting con Optuna.

Estrategia
----------
- Muestreador TPE (bayesiano) con semilla fija: reproducible.
- Validación **temporal** (holdout cronológico train -> val), no k-fold
  aleatorio, para no evaluar con el pasado un modelo entrenado con el futuro.
- Se minimiza el MAE de validación (misma métrica con la que se elige el
  campeón).
- Cada intento se puede registrar como corrida anidada de MLflow mediante el
  argumento ``al_terminar_intento``; así el módulo no depende de MLflow y se
  prueba de forma aislada.

El espacio de búsqueda se define en ``params.yaml`` (sección ``optuna``).
"""

from __future__ import annotations

from collections.abc import Callable

from loguru import logger
import optuna

from tc5044_mlops.config import cargar_params
from tc5044_mlops.modeling.evaluate import metricas_regresion
from tc5044_mlops.modeling.models import crear_modelo
from tc5044_mlops.pipeline import construir_pipeline

LOGARITMICOS = {"learning_rate", "l2_regularization"}
ENTEROS = {"max_iter", "max_leaf_nodes", "min_samples_leaf"}


def sugerir_hiperparametros(trial: optuna.Trial, espacio: dict) -> dict:
    """Traduce el espacio de params.yaml a sugerencias de Optuna."""
    hp = {}
    for nombre, (bajo, alto) in espacio.items():
        if nombre in ENTEROS:
            hp[nombre] = trial.suggest_int(nombre, int(bajo), int(alto))
        else:
            hp[nombre] = trial.suggest_float(nombre, float(bajo), float(alto),
                                             log=nombre in LOGARITMICOS)
    return hp


def optimizar_hgb(X_tr, y_tr, X_va, y_va, n_trials: int | None = None,
                  cfg: dict | None = None, semilla: int | None = None,
                  al_terminar_intento: Callable[[int, dict, dict], None] | None = None,
                  ) -> optuna.Study:
    """Busca los hiperparámetros del HGB que minimizan el MAE de validación.

    Parameters
    ----------
    al_terminar_intento
        Función opcional ``f(numero, hiperparametros, metricas)`` que se llama
        al final de cada intento (se usa para registrar en MLflow).
    """
    cfg = cfg or cargar_params("optuna")
    semilla = cargar_params("semilla") if semilla is None else semilla
    n_trials = cfg["n_trials"] if n_trials is None else n_trials

    if cfg.get("fraccion_train", 1.0) < 1.0:
        idx = X_tr.sample(frac=cfg["fraccion_train"], random_state=semilla).index
        X_tr, y_tr = X_tr.loc[idx], y_tr.loc[idx]
        logger.info(f"Optuna sobre una muestra de {len(X_tr):,} filas de train")

    def objetivo(trial: optuna.Trial) -> float:
        hp = sugerir_hiperparametros(trial, cfg["espacio"])
        pipe = construir_pipeline(crear_modelo("hgb", hp, semilla), semilla=semilla)
        pipe.fit(X_tr, y_tr)
        met = metricas_regresion(y_va, pipe.predict(X_va), prefijo="val_")
        for k, v in met.items():
            trial.set_user_attr(k, v)
        if al_terminar_intento:
            al_terminar_intento(trial.number, hp, met)
        return met["val_mae"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(direction="minimize", study_name="hgb-mibici",
                                  sampler=optuna.samplers.TPESampler(seed=semilla))
    estudio.optimize(objetivo, n_trials=n_trials, timeout=cfg.get("timeout_seg"))
    logger.success(f"Optuna: mejor val_mae={estudio.best_value:.4f} con {estudio.best_params}")
    return estudio
