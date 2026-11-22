# Arquitectura de la solución (pipeline de MLOps)

```mermaid
flowchart LR
    subgraph Datos["Datos · DVC"]
        A[MiBici datos abiertos<br/>CSV mensuales] -->|descargar| B[data/raw]
        B -->|consolidar| C[data/interim]
        C -->|limpiar| D[data/processed<br/>viajes limpios]
        D -->|features| E[tabla estación × hora]
    end
    subgraph Experimentos["Experimentos · MLflow"]
        E -->|train + Optuna| F[Tracking<br/>corridas y métricas]
        F --> G[Model Registry<br/>@champion v1]
    end
    subgraph Despliegue["Servicio · FastAPI + Podman"]
        G -->|exportar| H[models/champion<br/>modelo + golden set]
        H --> I[Imagen ml-service]
        I -->|push| J[DockerHub<br/>jhonatan0925/mibici-demanda-api]
        I --> K[POST /predict<br/>GET /health · /model-info]
    end
    subgraph Monitoreo["Monitoreo · drift"]
        E --> L[Escenarios sintéticos]
        H --> L
        L --> M[PSI · KS · nulos · MAE<br/>alertas y acciones]
        M -->|reentrenar| F
    end
    N[GitHub Actions<br/>pytest -q + humo del contenedor] -.-> I
```

| Componente | Herramienta | Archivos |
|---|---|---|
| Estructura del proyecto | Cookiecutter Data Science v2 | árbol del repositorio |
| Parámetros | `params.yaml` (fuente única) | `tc5044_mlops/config.py` |
| Versionado de datos y pipeline | DVC (remoto local) + Git | `dvc.yaml`, `dvc.lock` |
| Experimentos y registro de modelos | MLflow 3 (Tracking + Model Registry) | `tracking.py`, `modeling/train.py` |
| Orquestación | Makefile, DVC y MLflow Projects | `Makefile`, `MLproject` |
| Pruebas | pytest (unitarias + integración) | `tests/` |
| Servicio | FastAPI + Pydantic + uvicorn | `tc5044_mlops/api/` |
| Empaquetado | Podman / Docker (`python:3.11-slim`) | `Dockerfile`, `requirements-serving.txt` |
| Registro de imágenes | DockerHub | `jhonatan0925/mibici-demanda-api` |
| Reproducibilidad | golden set + re-entrenamiento con semilla | `reproducibilidad.py`, `verificar_servicio.py` |
| Monitoreo de drift | PSI/KS propios + Evidently | `tc5044_mlops/monitoring/` |
| Integración continua | GitHub Actions | `.github/workflows/ci.yml` |
