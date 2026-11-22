#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = tc5044-mlops
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

# Fase 3: contenedor (Podman por defecto; `make imagen MOTOR=docker` para Docker)
MOTOR ?= podman
IMAGEN_LOCAL ?= ml-service:latest
IMAGEN ?= docker.io/jhonatan0925/mibici-demanda-api
VERSION ?= 1.0.0
MODELO_V ?= 1
FORMATO = $(if $(filter podman,$(MOTOR)),--format docker,)

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv pip install -r requirements.txt -c constraints.txt
	uv pip install -e .
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests (unitarias + integración): un solo comando
.PHONY: test
test:
	python -m pytest -q


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Descargar los CSV de MiBici (rango de params.yaml)
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m tc5044_mlops.dataset descargar
	$(PYTHON_INTERPRETER) -m tc5044_mlops.dataset consolidar

## Control de calidad del dataset consolidado
.PHONY: calidad
calidad:
	$(PYTHON_INTERPRETER) -m tc5044_mlops.calidad

## Limpiar y enriquecer el dataset (umbrales en params.yaml)
.PHONY: limpiar
limpiar:
	$(PYTHON_INTERPRETER) -m tc5044_mlops.limpieza

## Construir la tabla de modelado estación × hora
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m tc5044_mlops.features

## Entrenar, ajustar (Optuna) y registrar los modelos en MLflow
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m tc5044_mlops.modeling.train

## Entrenamiento rápido sin Optuna (para revisar que todo corre)
.PHONY: train-rapido
train-rapido:
	$(PYTHON_INTERPRETER) -m tc5044_mlops.modeling.train --n-trials 0

## Predecir con el modelo campeón del Model Registry
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) -m tc5044_mlops.modeling.predict

## Levantar el servidor de MLflow (UI en http://127.0.0.1:5000)
.PHONY: mlflow-server
mlflow-server:
	mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

## Pipeline completo de datos a modelo registrado
.PHONY: all
all: data limpiar features train

#################################################################################
# FASE 3: SERVICIO, CONTENEDOR, REPRODUCIBILIDAD Y DRIFT                        #
#################################################################################

## Exportar el modelo @champion del Registry a models/champion (+ golden set)
.PHONY: exportar
exportar:
	$(PYTHON_INTERPRETER) -m tc5044_mlops.modeling.exportar

## Verificar que el modelo exportado reproduce el golden set en este entorno
.PHONY: verificar-modelo
verificar-modelo:
	ENTORNO=wsl-entrenamiento $(PYTHON_INTERPRETER) -m tc5044_mlops.reproducibilidad verificar

## Levantar la API en local con recarga (http://127.0.0.1:8000/docs)
.PHONY: api
api:
	uvicorn tc5044_mlops.api.main:app --reload --host 127.0.0.1 --port 8000

## Regenerar docs/openapi.json
.PHONY: openapi
openapi:
	$(PYTHON_INTERPRETER) -m tc5044_mlops.api.openapi

## Construir la imagen del servicio (requiere make exportar)
.PHONY: imagen
imagen:
	$(MOTOR) build $(FORMATO) -t $(IMAGEN_LOCAL) \
		--build-arg VERSION=$(VERSION) --build-arg GIT_COMMIT=$$(git rev-parse --short HEAD) .

## Correr el contenedor en segundo plano en el puerto 8000
.PHONY: contenedor
contenedor:
	$(MOTOR) run -d --rm --name mibici-api -p 8000:8000 $(IMAGEN_LOCAL)

## Detener el contenedor
.PHONY: contenedor-detener
contenedor-detener:
	$(MOTOR) stop mibici-api

## Verificar portabilidad: golden set por HTTP, dentro de la imagen y re-entrenamiento
.PHONY: verificar-contenedor
verificar-contenedor:
	mkdir -p reports/reproducibilidad
	$(PYTHON_INTERPRETER) -m tc5044_mlops.verificar_servicio --url http://127.0.0.1:8000
	$(MOTOR) run --rm $(IMAGEN_LOCAL) python -m tc5044_mlops.reproducibilidad verificar \
		--salida - > reports/reproducibilidad/verificacion_modelo_contenedor.json
	$(MOTOR) run --rm -v ./data/processed:/app/data/processed:ro,Z $(IMAGEN_LOCAL) \
		python -m tc5044_mlops.reproducibilidad reentrenar \
		--salida - > reports/reproducibilidad/reentrenamiento_contenedor.json

## Etiquetar y publicar la imagen en DockerHub (requiere $(MOTOR) login docker.io)
.PHONY: publicar
publicar:
	$(MOTOR) tag $(IMAGEN_LOCAL) $(IMAGEN):$(VERSION)-mv$(MODELO_V)
	$(MOTOR) tag $(IMAGEN_LOCAL) $(IMAGEN):$(VERSION)
	$(MOTOR) tag $(IMAGEN_LOCAL) $(IMAGEN):latest
	$(MOTOR) push $(IMAGEN):$(VERSION)-mv$(MODELO_V)
	$(MOTOR) push $(IMAGEN):$(VERSION)
	$(MOTOR) push $(IMAGEN):latest

## Simular data drift, evaluar el desempeño y emitir alertas
.PHONY: drift
drift:
	$(PYTHON_INTERPRETER) -m tc5044_mlops.monitoring.drift


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
