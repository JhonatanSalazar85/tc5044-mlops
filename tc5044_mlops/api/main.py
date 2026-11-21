"""Servicio FastAPI del modelo de demanda MiBici.

Endpoints:
    GET  /health      estado del servicio (200 si el modelo está cargado, 503 si no)
    GET  /model-info  ficha del modelo: URI del Registry, versión, métricas, SHA-256
    POST /predict     predicción por lote de demanda estación × hora
    GET  /docs        Swagger UI (OpenAPI) · /redoc · /openapi.json

El modelo se carga **una sola vez** al arrancar (``lifespan``) desde la carpeta
exportada (``MODELO_DIR``, por defecto ``models/champion``). Si la carga falla
el proceso sigue vivo y ``/health`` responde 503: así el orquestador (o el
HEALTHCHECK del contenedor) lo marca como no saludable en vez de reiniciarlo en
bucle sin dejar rastro del error.

Ejecución local:
    uvicorn tc5044_mlops.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import time

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from loguru import logger

from tc5044_mlops.api.esquemas import (
    InfoModelo,
    Prediccion,
    RespuestaPrediccion,
    Salud,
    SolicitudPrediccion,
)
from tc5044_mlops.api.servicio import ServicioModelo, ruta_modelo
from tc5044_mlops.config import cargar_params

_API = cargar_params("api")
VERSION_API = str(_API["version"])

DESCRIPCION = """
Predice cuántas bicicletas se **retirarán** de una estación de MiBici (Guadalajara) en una hora
dada, para que el operador pase de un rebalanceo reactivo a uno preventivo.

* Horizonte: **1 hora** (el modelo usa la demanda de la hora anterior).
* Modelo: `HistGradientBoostingRegressor` con pérdida Poisson dentro de un `Pipeline` de
  scikit-learn (imputación + escalado, `TargetEncoder` de la estación y PCA del perfil horario),
  registrado en MLflow como `mibici-demanda-estacion-hora` con alias `@champion`.
* Las variables de calendario las calcula el servicio a partir de `fecha_hora`.

Proyecto TC5044 Operaciones de aprendizaje automático (Tec de Monterrey), Equipo 15.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    directorio = ruta_modelo()
    app.state.servicio = None
    app.state.error_carga = None
    try:
        app.state.servicio = ServicioModelo(directorio)
    except Exception as exc:  # noqa: BLE001 - se informa en /health
        app.state.error_carga = f"{type(exc).__name__}: {exc}"
        logger.error(f"No se pudo cargar el modelo desde {directorio}: {app.state.error_carga}")
    yield


ERROR_503 = {503: {"description": "El modelo no está cargado."}}


def obtener_servicio(request: Request) -> ServicioModelo:
    """Dependencia de FastAPI: el modelo cargado o HTTP 503."""
    servicio = request.app.state.servicio
    if servicio is None:
        raise HTTPException(status_code=503,
                            detail=f"Modelo no disponible: {request.app.state.error_carga}")
    return servicio


def crear_app() -> FastAPI:
    """Construye la aplicación (fábrica: las pruebas crean instancias independientes)."""
    app = FastAPI(
        title=_API["titulo"],
        version=VERSION_API,
        description=DESCRIPCION,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "operación", "description": "Salud del servicio."},
            {"name": "modelo", "description": "Ficha del modelo desplegado."},
            {"name": "predicción", "description": "Inferencia de demanda estación × hora."},
        ],
    )

    @app.middleware("http")
    async def medir_tiempo(request: Request, call_next):
        inicio = time.perf_counter()
        respuesta = await call_next(request)
        ms = 1000 * (time.perf_counter() - inicio)
        respuesta.headers["X-Tiempo-Proceso-ms"] = f"{ms:.1f}"
        logger.info(f"{request.method} {request.url.path} -> {respuesta.status_code} "
                    f"({ms:.1f} ms)")
        return respuesta

    @app.get("/", include_in_schema=False)
    def raiz():
        return RedirectResponse(url="/docs")

    @app.get("/health", response_model=Salud, tags=["operación"], responses=ERROR_503)
    def health(request: Request):
        """200 con el modelo cargado; 503 si la carga falló (lo usa el HEALTHCHECK)."""
        servicio = request.app.state.servicio
        if servicio is None:
            raise HTTPException(status_code=503, detail=Salud(
                estado="sin_modelo", modelo_cargado=False, version_api=VERSION_API,
                detalle=request.app.state.error_carga).model_dump())
        return Salud(estado="ok", modelo_cargado=True, version_api=VERSION_API,
                     modelo=servicio.info["nombre"], version_modelo=str(servicio.info["version"]))

    @app.get("/model-info", response_model=InfoModelo, tags=["modelo"], responses=ERROR_503)
    def model_info(servicio: ServicioModelo = Depends(obtener_servicio)):
        """Ficha del modelo desplegado: URI del Registry, versión, métricas y SHA-256."""
        info = servicio.info
        return InfoModelo(
            nombre=info["nombre"], version=str(info["version"]), alias=info.get("alias"),
            model_uri=info["model_uri"], run_id=info.get("run_id"), modelo=info.get("modelo"),
            algoritmo=info.get("algoritmo"),
            metricas_validacion_prueba=info.get("metricas_exportacion") or {},
            hiperparametros=info.get("hiperparametros"), columnas_entrada=servicio.columnas,
            sha256_modelo=info.get("sha256_modelo"), git_commit=info.get("git_commit"),
            exportado_utc=info.get("exportado_utc"),
            entorno_exportacion=info.get("entorno_exportacion"))

    @app.post("/predict", response_model=RespuestaPrediccion, tags=["predicción"],
              responses={**ERROR_503, 422: {"description": "Entrada inválida (por campo)."}})
    def predict(solicitud: SolicitudPrediccion,
                servicio: ServicioModelo = Depends(obtener_servicio)):
        """Predice la demanda de retiros para cada registro (estación, hora, rezagos)."""
        try:
            salida, advertencias = servicio.predecir(solicitud.registros)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error al predecir")
            raise HTTPException(status_code=500, detail=f"Error al predecir: {exc}") from exc
        return RespuestaPrediccion(
            modelo=servicio.info["nombre"], version_modelo=str(servicio.info["version"]),
            model_uri=servicio.info["model_uri"], version_api=VERSION_API,
            n_registros=len(salida),
            predicciones=[Prediccion(origen_id=int(f.origen_id),
                                     fecha_hora=f.fecha_hora.to_pydatetime(),
                                     prediccion=float(f.prediccion))
                          for f in salida.itertuples(index=False)],
            advertencias=advertencias)

    return app


app = crear_app()
