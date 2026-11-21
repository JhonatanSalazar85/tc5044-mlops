"""Pruebas del empaquetado en contenedor y de la verificación del servicio.

Las pruebas estáticas revisan el Dockerfile, `.dockerignore` y las dependencias
fijadas sin necesidad de Podman/Docker (corren en la CI). La verificación del
servicio usa el `TestClient` de FastAPI como cliente HTTP: es el mismo código que
`make verificar-contenedor` ejecuta contra la imagen en marcha.
"""

from pathlib import Path
import re

from fastapi.testclient import TestClient
import pytest

from tc5044_mlops.api.main import crear_app
from tc5044_mlops.config import PROJ_ROOT, cargar_params
from tc5044_mlops.modeling import exportar
from tc5044_mlops.verificar_servicio import verificar_servicio

DOCKERFILE = (PROJ_ROOT / "Dockerfile").read_text(encoding="utf-8")


def _pins(ruta: Path) -> dict[str, str]:
    pins = {}
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.split("#")[0].strip()
        if linea:
            nombre, _, version = linea.partition("==")
            pins[nombre.lower()] = version
    return pins


# --------------------------------------------------------------------------
# Dockerfile y contexto de construcción
# --------------------------------------------------------------------------
def test_imagen_base_es_python_311_slim():
    assert re.search(r"ARG BASE_IMAGE=docker\.io/library/python:3\.11-slim", DOCKERFILE)


def test_el_servicio_no_corre_como_root():
    usuario = re.findall(r"^USER\s+(\S+)", DOCKERFILE, flags=re.M)
    assert usuario and usuario[-1] not in {"root", "0"}


def test_expone_el_puerto_healthcheck_y_arranca_uvicorn():
    assert re.search(r"^EXPOSE 8000$", DOCKERFILE, flags=re.M)
    assert "HEALTHCHECK" in DOCKERFILE and "/health" in DOCKERFILE
    assert '"uvicorn", "tc5044_mlops.api.main:app"' in DOCKERFILE


def test_la_imagen_contiene_el_modelo_y_sus_dependencias_exactas():
    assert "COPY models/champion ./models/champion" in DOCKERFILE
    assert "models/champion/requirements.txt" in DOCKERFILE


def test_dockerignore_excluye_datos_y_secretos():
    reglas = (PROJ_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for regla in ("data/", "mlflow.db", ".env", ".venv", ".git", "mlruns/"):
        assert regla in reglas, regla
    assert "!models/champion" in reglas


def test_dependencias_del_servicio_fijadas_y_coherentes_con_requirements():
    servicio = _pins(PROJ_ROOT / "requirements-serving.txt")
    proyecto = _pins(PROJ_ROOT / "requirements.txt")
    assert servicio and all(servicio.values()), "toda dependencia debe fijarse con =="
    for paquete, version in servicio.items():
        assert proyecto.get(paquete) == version, f"{paquete}: {version} != {proyecto.get(paquete)}"


# --------------------------------------------------------------------------
# Verificación del servicio contra el golden set
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def modelo_dir(tmp_path_factory):
    destino = tmp_path_factory.mktemp("cont") / "champion"
    exportar.exportar_sintetico(destino=destino)
    return destino


def test_verificar_servicio_reproduce_el_golden_set(modelo_dir, monkeypatch):
    monkeypatch.setenv("MODELO_DIR", str(modelo_dir))
    with TestClient(crear_app()) as cliente:
        resultado = verificar_servicio(cliente, modelo_dir, lote=400, peticiones_unitarias=3)
    assert resultado["mismo_artefacto"]["coincide"]
    assert resultado["comparacion"]["identicas_bit_a_bit"]
    assert resultado["metricas_servicio"]["mae"] == pytest.approx(
        resultado["metricas_esperadas"]["mae"], abs=1e-12)
    assert resultado["latencia_un_registro"]["n"] == 3


def test_verificar_servicio_detecta_otro_artefacto(modelo_dir, tmp_path, monkeypatch):
    """Si el servicio sirve un modelo distinto al local, la verificación lo reporta."""
    otro = tmp_path / "otro"
    params = cargar_params()
    params["semilla"] = 7
    exportar.exportar_sintetico(params=params, destino=otro)
    monkeypatch.setenv("MODELO_DIR", str(otro))
    with TestClient(crear_app()) as cliente:
        resultado = verificar_servicio(cliente, modelo_dir, peticiones_unitarias=0)
    assert not resultado["mismo_artefacto"]["coincide"]
