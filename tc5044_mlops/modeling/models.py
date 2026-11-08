"""Catálogo de modelos candidatos y su justificación.

El problema es una regresión sobre conteos (bicicletas retiradas por estación
y hora) con fuerte estacionalidad diaria y semanal, muchos ceros y relaciones
no lineales entre hora, tipo de estación y demanda reciente. Se compara una
escalera de complejidad creciente; cada peldaño debe justificar su costo:

| Nombre          | Algoritmo                           | Por qué se incluye                          |
|-----------------|-------------------------------------|---------------------------------------------|
| baseline        | misma hora de la semana anterior    | piso de referencia sin aprendizaje          |
| ridge           | regresión lineal con penalización L2| modelo lineal interpretable; mide cuánto    |
|                 |                                     | explican los rezagos y el calendario solos  |
| poisson         | GLM de Poisson (liga log)           | supuesto estadístico correcto para conteos: |
|                 |                                     | no predice negativos y el efecto es         |
|                 |                                     | multiplicativo (hora punta × estación)      |
| random_forest   | bosque aleatorio                    | no lineal, robusto, poca sintonía; captura  |
|                 |                                     | interacciones hora × estación               |
| hgb             | HistGradientBoosting, pérdida       | estado del arte en tabulares grandes, rápido|
|                 | Poisson                             | con ~700 k filas, maneja nulos; es el que   |
|                 |                                     | se ajusta con Optuna                        |

Los hiperparámetros de partida viven en ``params.yaml`` (sección ``modelos``).
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor, Ridge

from tc5044_mlops.config import cargar_params
from tc5044_mlops.modeling.evaluate import BaselineEstacional

JUSTIFICACION = {
    "baseline": "Regla ingenua estacional (lag 168 h): piso que todo modelo debe superar.",
    "ridge": "Lineal con L2: interpretable; mide el aporte de rezagos y calendario.",
    "poisson": "GLM Poisson: supuesto adecuado para conteos, sin predicciones negativas.",
    "random_forest": "Bosque aleatorio: no lineal, captura interacciones hora × estación.",
    "hgb": "Gradient boosting con pérdida Poisson: mejor relación precisión/tiempo en tabulares.",
}

# Los modelos que no usan el preprocesador (leen columnas crudas)
SIN_PREPROCESADOR = {"baseline"}


def nombres_modelos() -> list[str]:
    return list(JUSTIFICACION)


def crear_modelo(nombre: str, params: dict | None = None, semilla: int | None = None):
    """Instancia el estimador `nombre` con los hiperparámetros de params.yaml.

    ``params`` sobrescribe (por ejemplo, con los mejores valores de Optuna).
    """
    if nombre not in JUSTIFICACION:
        raise ValueError(f"Modelo desconocido '{nombre}'. Opciones: {nombres_modelos()}")
    semilla = cargar_params("semilla") if semilla is None else semilla
    hp = cargar_params("modelos").get(nombre, {}) | (params or {})

    if nombre == "baseline":
        return BaselineEstacional(**hp)
    if nombre == "ridge":
        return Ridge(**hp, random_state=semilla)
    if nombre == "poisson":
        return PoissonRegressor(**hp)
    if nombre == "random_forest":
        return RandomForestRegressor(**hp, n_jobs=-1, random_state=semilla)
    return HistGradientBoostingRegressor(**hp, random_state=semilla)
