# tc5044-mlops — Demanda de MiBici por estación y hora

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto del curso TC5044 Operaciones de aprendizaje automático (Tec de Monterrey), Equipo 15.

**Problema:** predecir cuántas bicicletas se retirarán de cada estación de MiBici (Guadalajara) en
cada hora, para pasar de un rebalanceo reactivo a uno preventivo. Es una regresión sobre conteos
con datos abiertos de https://www.mibici.net/es/datos-abiertos/ (2025-01 a 2026-08).

## Organización del proyecto

Sigue la plantilla [Cookiecutter Data Science v2](https://cookiecutter-data-science.drivendata.org/).
Toda la lógica vive en el paquete `tc5044_mlops`; los notebooks solo importan sus módulos.

```
├── LICENSE
├── Makefile             <- Comandos del proyecto: `make data`, `make train`, `make test`...
├── MLproject            <- Puntos de entrada para MLflow Projects (`mlflow run`)
├── python_env.yaml      <- Entorno de MLflow Projects
├── README.md
├── params.yaml          <- ÚNICA fuente de parámetros: rutas, umbrales, cortes, hiperparámetros, semillas
├── dvc.yaml             <- Pipeline reproducible: descargar → consolidar → limpiar → features → train
├── .env.example         <- Variables de entorno (MLFLOW_TRACKING_URI); copiar como .env
├── data                 <- No se versiona en Git (ver .gitignore)
│   ├── external
│   ├── interim          <- Viajes consolidados (unión fiel de los CSV)
│   ├── processed        <- Viajes limpios y tabla de modelado estación × hora
│   └── raw              <- CSV originales de MiBici (inmutables)
├── docs                 <- Proyecto mkdocs y ML Canvas
├── models               <- Los modelos se guardan en MLflow; esta carpeta queda para exportaciones
├── notebooks            <- Exploración y comunicación; importan el paquete
│   ├── 0.01-jms-eda-mibici.ipynb
│   └── 3.01-jms-modelado-mibici.ipynb
├── references           <- Diccionarios de datos y material de referencia
├── reports
│   ├── figures          <- Figuras del EDA y de la evaluación de modelos
│   ├── limpieza.json    <- Métricas de la etapa de limpieza
│   └── metricas_fase2.json <- Métricas de los modelos (generado por `make train`)
├── requirements.txt     <- Dependencias fijadas
├── constraints.txt      <- Restricciones de compatibilidad (scikit-learn<1.8, setuptools<81)
├── pyproject.toml       <- Metadatos del paquete y configuración de ruff
├── tests                <- Pruebas con pytest (datos sintéticos, sin descargas)
└── tc5044_mlops         <- Código fuente del proyecto
    ├── __init__.py
    ├── config.py        <- Rutas del proyecto y carga de params.yaml
    ├── dataset.py       <- Descarga y consolidación de los CSV
    ├── calidad.py       <- Control de calidad del dataset consolidado
    ├── limpieza.py      <- Limpieza: marcar (no borrar) viajes inválidos
    ├── eda.py           <- Funciones del análisis exploratorio
    ├── features.py      <- Tabla estación × hora: calendario cíclico, rezagos, medias móviles
    ├── pipeline.py      <- Pipeline de scikit-learn: escalado, TargetEncoder, PCA del perfil horario
    ├── tracking.py      <- MLflow: configuración, trazabilidad (SHA-256) y Model Registry
    ├── versionado.py    <- Versionado de datasets en MLflow
    ├── plots.py         <- Figuras de evaluación
    └── modeling
        ├── __init__.py
        ├── models.py    <- Catálogo de modelos candidatos y su justificación
        ├── evaluate.py  <- Métricas (MAE, RMSE, WAPE, devianza Poisson, R²) y línea base
        ├── tune.py      <- Búsqueda de hiperparámetros con Optuna
        ├── train.py     <- Entrenamiento, registro en MLflow y selección del campeón
        └── predict.py   <- Inferencia con el modelo @champion del Model Registry
```

## Instalación del entorno

Requisitos: WSL2/Linux, [uv](https://docs.astral.sh/uv/), Python 3.11.

```bash
make create_environment          # uv venv --python 3.11
source .venv/bin/activate
make requirements                # dependencias fijadas + paquete en modo editable
make test                        # pytest
```

`constraints.txt` fija `scikit-learn<1.8` y `setuptools<81` por compatibilidad con Deepchecks 0.19.1.

## Reproducir el proyecto

```bash
make data          # descarga y consolida los CSV (rango en params.yaml)
make calidad       # diagnóstico de calidad
make limpiar       # data/processed/mibici_viajes.parquet + reports/limpieza.json
make features      # data/processed/mibici_demanda_horaria.parquet
make mlflow-server # en otra terminal: UI en http://127.0.0.1:5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
make train         # entrena, ajusta con Optuna y registra los modelos
make predict       # predicciones del modelo @champion
```

Alternativas equivalentes: `dvc repro` (re-ejecuta solo las etapas cuyos datos, código o
parámetros cambiaron) o MLflow Projects:

```bash
mlflow run . -e train --env-manager local --experiment-name mibici-demanda-fase2 -P n_trials=30
```

Para cambiar un umbral, un corte de partición o un hiperparámetro se edita `params.yaml`; ningún
valor configurable está escrito en el código.

## Pipeline de modelado

1. **Tabla de modelado** (`features.py`): rejilla estación × hora con ceros explícitos; seno/coseno
   de hora, día y mes; rezagos de 1, 2, 24 y 168 h; medias móviles de 3, 24 y 168 h calculadas por
   estación y desplazadas una hora para evitar fuga de información.
2. **Partición cronológica**: train < 2026-05-01 ≤ validación < 2026-07-01 ≤ prueba.
3. **Preprocesamiento** (`pipeline.py`, un solo `Pipeline` de scikit-learn):
   imputación + `StandardScaler` en numéricas; `TargetEncoder` para la estación;
   `PerfilHorarioPCA` (PCA del perfil horario de cada estación, que resume su tipo de uso).
4. **Modelos** (`modeling/models.py`): línea base estacional, Ridge, GLM Poisson, Random Forest y
   HistGradientBoosting con pérdida Poisson; este último se ajusta con Optuna (TPE).
5. **Evaluación** (`modeling/evaluate.py`): MAE (métrica de selección), RMSE, WAPE, devianza de
   Poisson y R². Se elige con validación; prueba solo estima el desempeño fuera de muestra.
6. **MLflow** (`modeling/train.py`, `tracking.py`): una corrida padre por ejecución, una hija por
   modelo y una nieta por intento de Optuna. Cada corrida guarda hiperparámetros, métricas,
   figuras, el SHA-256 de los datos, el commit de Git y el modelo con su firma. Los modelos se
   registran en `mibici-demanda-estacion-hora` con alias `@champion` y `@challenger`.
