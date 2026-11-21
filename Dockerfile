# Imagen del servicio de predicción de demanda MiBici (TC5044 MLOps, Equipo 15).
#
# Requisito previo: `make exportar` (crea models/champion desde el Model Registry).
#
# Construir y correr (desde la raíz del repositorio):
#   podman build --format docker -t ml-service:latest .
#   podman run --rm -p 8000:8000 ml-service:latest
# Con Docker los comandos son los mismos cambiando `podman` por `docker`
# (`--format docker` solo aplica a Podman: conserva el HEALTHCHECK, que el formato OCI ignora).

ARG BASE_IMAGE=docker.io/library/python:3.11-slim
FROM ${BASE_IMAGE}

ARG VERSION=1.0.0
ARG GIT_COMMIT=desconocido
ARG MODEL_URI=models:/mibici-demanda-estacion-hora/1
LABEL org.opencontainers.image.title="mibici-demanda-api" \
      org.opencontainers.image.description="FastAPI con el modelo de demanda MiBici por estación y hora" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.source="https://github.com/JhonatanSalazar85/tc5044-mlops" \
      org.opencontainers.image.licenses="MIT" \
      mx.tec.tc5044.model-uri="${MODEL_URI}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    MODELO_DIR=/app/models/champion \
    ENTORNO=contenedor \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# 1) Dependencias (capa en caché mientras no cambien los requirements).
#    Las del modelo vienen de MLflow con las versiones exactas del entrenamiento;
#    mlflow se cambia por mlflow-skinny de la misma versión.
COPY models/champion/requirements.txt /tmp/requirements-modelo.txt
COPY requirements-serving.txt /tmp/requirements-serving.txt
RUN sed -E 's/^mlflow==/mlflow-skinny==/' /tmp/requirements-modelo.txt > /tmp/requirements-skinny.txt \
 && pip install -r /tmp/requirements-skinny.txt -r /tmp/requirements-serving.txt \
 && rm -f /tmp/requirements-*.txt

# 2) Código y configuración (params.yaml es la fuente única de parámetros).
COPY params.yaml pyproject.toml LICENSE ./
COPY tc5044_mlops ./tc5044_mlops

# 3) Modelo exportado con su ficha (model_info.json) y su golden set (referencia.parquet).
COPY models/champion ./models/champion

# 4) Usuario sin privilegios; los archivos quedan de solo lectura para él.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "uvicorn", "tc5044_mlops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
