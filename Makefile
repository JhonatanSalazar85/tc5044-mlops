#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = tc5044-mlops
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

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



## Run tests
.PHONY: test
test:
	python -m pytest tests


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
