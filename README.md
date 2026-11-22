# tc5044-mlops — Demanda de MiBici por estación y hora

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto del curso TC5044 Operaciones de aprendizaje automático (Tec de Monterrey), Equipo 15:
Pedro Cuauhtémoc Márquez Correo, Fermín Moctezuma Domínguez, Irvin Uriel Mundo Rivera y
Jhonatan Marín Salazar.

**Problema:** predecir cuántas bicicletas se retirarán de cada estación de MiBici (Guadalajara)
en la próxima hora, para pasar de un rebalanceo reactivo a uno preventivo. Regresión sobre
conteos con datos abiertos de https://www.mibici.net/es/datos-abiertos/ (2025-01 a 2026-08).
ML Canvas completo: [`docs/ml_canvas.md`](docs/ml_canvas.md). Arquitectura:
[`docs/arquitectura.md`](docs/arquitectura.md).

## Resumen de la entrega final (Fase 3)

| Artefacto | Dónde |
|---|---|
| Modelo desplegado | `models:/mibici-demanda-estacion-hora/1` (alias `@champion`, hgb_optuna) |
| Copia exportada que usa el servicio | `models/champion/` (versionada con DVC, etapa `exportar`) |
| API | FastAPI, `POST /predict`, `GET /health`, `GET /model-info` — [`docs/api.md`](docs/api.md) |
| Esquema OpenAPI / Postman | [`docs/openapi.json`](docs/openapi.json), [`docs/postman_mibici_api.json`](docs/postman_mibici_api.json) |
| Imagen | `docker.io/jhonatan0925/mibici-demanda-api:1.0.0-mv1` — [`docs/contenedor.md`](docs/contenedor.md) |
| Pruebas | `pytest -q` (unitarias + integración) |
| Reproducibilidad | `reports/reproducibilidad/*.json` |
| Data drift | `reports/drift/` y figuras `reports/figures/fase3_drift_*.png` |

## Organización del proyecto

Sigue la plantilla [Cookiecutter Data Science v2](https://cookiecutter-data-science.drivendata.org/).
Toda la lógica vive en el paquete `tc5044_mlops`; los notebooks solo importan sus módulos.

```
├── Dockerfile             <- Imagen del servicio (Podman/Docker)
├── .dockerignore
├── Makefile               <- `make test`, `make exportar`, `make imagen`, `make drift`...
├── MLproject              <- Puntos de entrada para MLflow Projects (`mlflow run`)
├── params.yaml            <- ÚNICA fuente de parámetros (rutas, umbrales, semillas, drift)
├── dvc.yaml / dvc.lock    <- Pipeline reproducible: descargar → ... → train → exportar → drift
├── requirements.txt       <- Dependencias fijadas del proyecto
├── requirements-serving.txt <- Dependencias del servicio (además de las del modelo)
├── constraints.txt
├── data                   <- No se versiona en Git (DVC)
├── docs                   <- ML Canvas, arquitectura, API, contenedor, OpenAPI, Postman
├── models/champion        <- Modelo exportado del Registry (DVC)
├── notebooks
├── reports
│   ├── figures            <- EDA, modelos (fase2_*) y drift (fase3_drift_*)
│   ├── drift              <- Resultados de la simulación de drift
│   └── reproducibilidad   <- Verificaciones en WSL y en el contenedor
├── tests                  <- pytest: unitarias e integración (datos sintéticos)
└── tc5044_mlops
    ├── config.py          <- Rutas y carga de params.yaml
    ├── dataset.py, calidad.py, limpieza.py, eda.py, features.py
    ├── pipeline.py        <- Pipeline de scikit-learn (escalado, TargetEncoder, PCA del perfil)
    ├── tracking.py, versionado.py
    ├── sintetico.py       <- Datos sintéticos con la forma de MiBici (pruebas y CI)
    ├── reproducibilidad.py <- Semillas, huella del entorno, verificación y re-entrenamiento
    ├── verificar_servicio.py <- Verifica un servicio en marcha contra el golden set
    ├── modeling
    │   ├── models.py, evaluate.py, tune.py, train.py, predict.py
    │   └── exportar.py    <- Registry -> models/champion (+ model_info.json y golden set)
    ├── api
    │   ├── esquemas.py    <- Pydantic (entrada/salida)
    │   ├── servicio.py    <- Carga del modelo y preparación de variables
    │   ├── main.py        <- Endpoints FastAPI
    │   └── openapi.py     <- Exporta docs/openapi.json
    └── monitoring
        ├── estadisticos.py <- PSI, KS, nulos, consistencia de rezagos
        ├── escenarios.py   <- Escenarios sintéticos de drift
        └── drift.py        <- Evaluación, umbrales, alertas, figuras, Evidently, MLflow
```

## Instalación del entorno

Requisitos: WSL2/Linux, [uv](https://docs.astral.sh/uv/), Python 3.11 y Podman (o Docker).

```bash
make create_environment          # uv venv --python 3.11
source .venv/bin/activate
make requirements                # dependencias fijadas + paquete en modo editable
```

## Pruebas (un solo comando)

```bash
pytest -q          # o: make test
```

Las pruebas usan datos sintéticos (`tc5044_mlops/sintetico.py`) y bases SQLite temporales: no
descargan nada ni tocan `mlflow.db`. Cubren:

| Tipo | Archivos | Qué validan |
|---|---|---|
| Unitarias | `test_features`, `test_limpieza`, `test_pipeline`, `test_models`, `test_config`, `test_reproducibilidad`, `test_drift` | rezagos sin fuga, limpieza, preprocesamiento, métricas, semillas, PSI/KS y reglas de alerta |
| Integración | `test_integracion` | CSV crudos → consolidar → limpiar → features → pipeline → predicción → métricas → modelo serializado |
| Integración MLflow | `test_train`, `test_exportar` | corridas, Registry, alias, exportación y re-entrenamiento reproducible |
| Servicio | `test_api`, `test_contenedor` | validación Pydantic (422), paridad bit a bit API vs pipeline, Dockerfile, golden set por HTTP |

La CI (`.github/workflows/ci.yml`) corre `pytest -q` en cada PR y, además, construye la imagen con
un modelo sintético, la levanta y verifica `/health` y el golden set.

## Reproducir el pipeline

```bash
make data && make calidad && make limpiar && make features
make mlflow-server                 # en otra terminal: UI en http://127.0.0.1:5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
make train                         # entrena, ajusta con Optuna y registra (@champion)
make exportar                      # models:/mibici-demanda-estacion-hora@champion -> models/champion
make drift                         # simulación de drift y alertas
```

Equivalente con DVC (re-ejecuta solo lo que cambió): `dvc repro`. Con MLflow Projects:
`mlflow run . -e train --env-manager local --experiment-name mibici-demanda-fase2 -P n_trials=30`.

### Versionado de datos y artefactos (DVC + MLflow)

`dvc.lock` fija el hash de cada entrada y salida del pipeline (CSV crudos, parquet consolidado,
limpio, tabla de modelado y `models/champion`). Los archivos se guardan en un remoto DVC local:

```bash
dvc pull                    # recupera datos y modelo de la versión del commit actual
dvc status                  # qué etapas cambiaron
dvc metrics show            # reports/limpieza.json, metricas_fase2.json, drift/metricas_drift.json
```

Además, `make exportar` registra en MLflow una corrida `fase3-exportacion-v1` con
`model_info.json` y el golden set (`referencia.parquet`) como artefactos y el SHA-256 del dataset.

## Modelo registrado y servicio

| Concepto | Valor |
|---|---|
| Modelo registrado | `mibici-demanda-estacion-hora` |
| Versión servida | **1** (`models:/mibici-demanda-estacion-hora/1`, alias `@champion`) |
| Algoritmo | HistGradientBoosting (pérdida Poisson) ajustado con Optuna |
| Métricas | MAE 1.334 validación / 1.441 prueba; WAPE 0.41 y R² 0.74 en prueba |

```bash
make api                           # http://127.0.0.1:8000/docs
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"registros":[{"origen_id":51,"fecha_hora":"2026-08-31T08:00:00","demanda_lag_1h":18,
       "demanda_lag_2h":9,"demanda_lag_24h":31,"demanda_lag_168h":28,"demanda_media_3h":10.33,
       "demanda_media_24h":11.21,"demanda_media_168h":10.64}]}'
```

Esquema de entrada y salida, errores y ejemplos: [`docs/api.md`](docs/api.md).

## Contenedor

```bash
make exportar                                         # una vez
podman build --format docker -t ml-service:latest .   # o: docker build -t ml-service:latest .
podman run --rm -p 8000:8000 ml-service:latest        # o: docker run -p 8000:8000 ml-service:latest
```

Imagen publicada: **`docker.io/jhonatan0925/mibici-demanda-api`**

| Etiqueta | Contenido |
|---|---|
| `1.0.0-mv1` | servicio 1.0.0 + modelo v1 del Registry (inmutable) |
| `1.0.0` | versión del servicio |
| `latest` | última publicada |

```bash
podman run --rm -p 8000:8000 docker.io/jhonatan0925/mibici-demanda-api:1.0.0-mv1
```

Detalles de diseño, publicación y problemas frecuentes: [`docs/contenedor.md`](docs/contenedor.md).

## Reproducibilidad en otro entorno

- Dependencias fijadas (`requirements.txt`; el modelo lleva su propio `requirements.txt` generado
  por MLflow y la imagen lo instala tal cual) y semilla única (`semilla: 42` en `params.yaml`,
  aplicada a numpy, `TargetEncoder`, HGB y Optuna).
- `make verificar-modelo` (WSL) y `make verificar-contenedor` (imagen) comparan fila por fila el
  golden set: predicciones vía HTTP, dentro de la imagen y tras re-entrenar con la misma semilla.

## Data drift

`make drift` compara contra el conjunto de validación un control real (el periodo de prueba sin
alterar) y cuatro conjuntos sintéticos derivados de validación (desplazamiento de medias, variables
faltantes, cambio estacional y crecimiento de la demanda), mide PSI/KS, nulos,
consistencia de rezagos y MAE, y aplica umbrales (`params.yaml`, sección `drift`): PSI 0.1/0.2,
MAE +5 %/+15 %. Cada escenario termina en un nivel (OK/AVISO/ALERTA) y una acción (vigilar,
revisar el feature pipeline o reentrenar). Resultados en `reports/drift/alertas.md` y en MLflow
(experimento `mibici-monitoreo-fase3`).

## Roles (MLOps)

| Rol | Fase 3 |
|---|---|
| Data Engineer | DVC (`dvc.lock`, remoto), golden set, escenarios de datos de drift, consistencia de rezagos |
| Data Scientist | Métricas de referencia, umbrales PSI/MAE, análisis de escenarios y severidad |
| ML Engineer | Exportación del Registry, semillas, re-entrenamiento reproducible, versionado del modelo |
| Software Engineer | FastAPI + Pydantic, pruebas unitarias e integración, CI |
| Site Reliability Engineer | Imagen, HEALTHCHECK, usuario sin privilegios, publicación en DockerHub, alertas |
