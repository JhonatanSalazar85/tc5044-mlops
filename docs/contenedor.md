# Contenedor del servicio (Podman / Docker)

La imagen empaqueta el servicio FastAPI (`tc5044_mlops/api`), el modelo exportado
(`models/champion`, que corresponde a `models:/mibici-demanda-estacion-hora/1`, alias
`@champion`) y sus dependencias exactas. El equipo usa **Podman** en WSL (Ubuntu 24.04); los
comandos son idénticos con Docker cambiando `podman` por `docker`.

## Diseño de la imagen (`Dockerfile`)

| Decisión | Motivo |
|---|---|
| Base `docker.io/library/python:3.11-slim` (nombre completo) | Misma versión de Python que el entrenamiento; Podman no pregunta por *short names*. |
| Dependencias = `models/champion/requirements.txt` + `requirements-serving.txt` | MLflow escribió las versiones exactas del entorno de entrenamiento; el servicio solo agrega FastAPI/uvicorn. |
| `mlflow` → `mlflow-skinny` (misma versión) | Mismo código de carga de modelos sin servidor ni UI (cientos de MB menos). |
| Capa de dependencias antes que el código | Cambiar el código no reinstala paquetes (caché de capas). |
| Usuario `10001` sin privilegios, archivos de solo lectura | Principio de mínimo privilegio; la imagen puede correr con `--read-only`. |
| `HEALTHCHECK` contra `/health` | El motor marca el contenedor `healthy`/`unhealthy`; `/health` responde 503 si el modelo no cargó. |
| Etiquetas OCI (`version`, `revision`, `model-uri`) | Trazabilidad: de la imagen al commit y a la versión del Model Registry. |
| `.dockerignore` | El contexto no incluye datos, `mlflow.db`, `.env`, `.venv` ni `.git`. |

## Construir y correr

Requisito: haber corrido `make exportar` (crea `models/champion`).

```bash
podman build --format docker -t ml-service:latest \
  --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) .
podman run --rm -p 8000:8000 ml-service:latest
# equivalente con Docker:
docker build -t ml-service:latest . && docker run -p 8000:8000 ml-service:latest
```

`--format docker` solo aplica a Podman: el formato OCI por defecto descarta el `HEALTHCHECK`.

Con el contenedor arriba: <http://127.0.0.1:8000/docs> (también desde el navegador de Windows).

```bash
podman ps                 # STATUS debe decir (healthy) a los ~40 s
curl -s http://127.0.0.1:8000/health
```

## Verificar la portabilidad (otro entorno)

```bash
# 1) Desde el host: envía el golden set por HTTP y compara predicciones, métricas y SHA-256
python -m tc5044_mlops.verificar_servicio --url http://127.0.0.1:8000
#    -> reports/reproducibilidad/verificacion_servicio_contenedor.json

# 2) Dentro de la imagen: el modelo predice el golden set sin pasar por HTTP
podman run --rm ml-service:latest python -m tc5044_mlops.reproducibilidad verificar --salida - \
  > reports/reproducibilidad/verificacion_modelo_contenedor.json

# 3) Dentro de la imagen: re-entrenar un clon del modelo con la misma semilla
podman run --rm -v ./data/processed:/app/data/processed:ro,Z ml-service:latest \
  python -m tc5044_mlops.reproducibilidad reentrenar --salida - \
  > reports/reproducibilidad/reentrenamiento_contenedor.json
```

## Publicar en DockerHub

Repositorio: **`docker.io/jhonatan0925/mibici-demanda-api`**.

| Etiqueta | Significado | ¿Se mueve? |
|---|---|---|
| `1.0.0-mv1` | Servicio 1.0.0 + modelo versión 1 del Registry | No (inmutable) |
| `1.0.0` | Versión del servicio (`api.version` de `params.yaml`) | No |
| `latest` | Última imagen publicada | Sí |

Regla: un cambio de código sube la versión semántica del servicio (`1.0.1`, `1.1.0`); un
modelo nuevo en `@champion` cambia el sufijo `-mvN`.

```bash
podman login docker.io -u jhonatan0925          # contraseña = token de acceso de DockerHub
IMG=docker.io/jhonatan0925/mibici-demanda-api
for t in 1.0.0-mv1 1.0.0 latest; do podman tag ml-service:latest $IMG:$t; done
for t in 1.0.0-mv1 1.0.0 latest; do podman push $IMG:$t; done
# cualquier integrante del equipo:
podman run --rm -p 8000:8000 $IMG:1.0.0-mv1
```

## Problemas frecuentes

| Síntoma | Solución |
|---|---|
| `COPY models/champion...: no such file` | Falta `make exportar` (o `dvc pull models/champion`). |
| `HEALTHCHECK is not supported for OCI image format` | Construir con `--format docker`. |
| `short-name resolution enforced` | Usar el nombre completo `docker.io/library/...` (ya está en el Dockerfile). |
| El contenedor queda `unhealthy` | `podman logs <id>`: `/health` muestra el error de carga del modelo. |
