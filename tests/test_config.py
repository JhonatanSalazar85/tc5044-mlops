"""El archivo de parámetros es la fuente única de valores configurables."""

from tc5044_mlops import config


def test_params_tiene_las_secciones_del_pipeline():
    params = config.cargar_params()
    for seccion in ("semilla", "limpiar", "features", "particion", "pipeline",
                    "modelos", "optuna", "mlflow"):
        assert seccion in params, f"falta la sección {seccion} en params.yaml"


def test_cargar_seccion_devuelve_copia():
    a = config.cargar_params("modelos")
    a["ridge"]["alpha"] = -1
    assert config.cargar_params("modelos")["ridge"]["alpha"] != -1


def test_particion_es_cronologica():
    p = config.cargar_params("particion")
    assert p["corte_val"] < p["corte_test"]


def test_tracking_uri_respeta_variable_de_entorno(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    assert config.tracking_uri() == "http://127.0.0.1:5000"


def test_tracking_uri_sqlite_relativa_se_ancla_a_la_raiz(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = config.tracking_uri({"mlflow": {"tracking_uri": "sqlite:///mlflow.db"}})
    assert uri == f"sqlite:///{config.PROJ_ROOT / 'mlflow.db'}"
