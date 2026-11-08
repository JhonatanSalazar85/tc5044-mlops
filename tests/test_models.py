"""Pruebas de métricas, línea base, catálogo de modelos y búsqueda con Optuna."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from tc5044_mlops import plots
from tc5044_mlops.features import particion_temporal
from tc5044_mlops.modeling.evaluate import BaselineEstacional, metricas_regresion
from tc5044_mlops.modeling.models import SIN_PREPROCESADOR, crear_modelo, nombres_modelos
from tc5044_mlops.modeling.tune import optimizar_hgb
from tc5044_mlops.pipeline import columnas_entrada, construir_pipeline


@pytest.fixture(scope="module")
def particion(tabla_sintetica, cortes_sinteticos):
    tr, va, _ = particion_temporal(tabla_sintetica, **cortes_sinteticos)
    cols = columnas_entrada()
    return tr[cols], tr["demanda"], va[cols], va["demanda"]


def test_metricas_en_caso_perfecto():
    m = metricas_regresion([0, 2, 4], [0, 2, 4], prefijo="val_")
    assert m["val_mae"] == 0 and m["val_rmse"] == 0 and m["val_wape"] == 0
    assert m["val_r2"] == 1


def test_metricas_recortan_predicciones_negativas():
    assert metricas_regresion([0, 0], [-3, -1])["mae"] == 0


def test_wape_no_falla_con_ceros_en_el_objetivo():
    m = metricas_regresion([0, 0, 10], [1, 0, 9])
    assert m["wape"] == pytest.approx(2 / 10)


def test_baseline_usa_el_rezago_semanal_y_su_respaldo():
    X = pd.DataFrame({"demanda_lag_168h": [5.0, np.nan], "demanda_media_24h": [1.0, 3.0]})
    pred = BaselineEstacional().fit(X, [0, 0]).predict(X)
    assert pred.tolist() == [5.0, 3.0]


@pytest.mark.parametrize("nombre", nombres_modelos())
def test_cada_modelo_del_catalogo_entrena(nombre, particion):
    X_tr, y_tr, X_va, y_va = particion
    pipe = construir_pipeline(crear_modelo(nombre), preprocesar=nombre not in SIN_PREPROCESADOR)
    pipe.fit(X_tr, y_tr)
    m = metricas_regresion(y_va, pipe.predict(X_va))
    assert np.isfinite(m["mae"])


def test_modelos_aprendidos_superan_a_la_linea_base(particion):
    X_tr, y_tr, X_va, y_va = particion
    base = construir_pipeline(crear_modelo("baseline"), preprocesar=False).fit(X_tr, y_tr)
    hgb = construir_pipeline(crear_modelo("hgb")).fit(X_tr, y_tr)
    assert (metricas_regresion(y_va, hgb.predict(X_va))["mae"]
            < metricas_regresion(y_va, base.predict(X_va))["mae"])


def test_optuna_registra_cada_intento_y_es_reproducible(particion):
    X_tr, y_tr, X_va, y_va = particion
    cfg = {"n_trials": 3, "timeout_seg": None, "fraccion_train": 1.0,
           "espacio": {"learning_rate": [0.05, 0.2], "max_iter": [20, 40]}}
    intentos = []
    e1 = optimizar_hgb(X_tr, y_tr, X_va, y_va, cfg=cfg, semilla=1,
                       al_terminar_intento=lambda n, hp, m: intentos.append((n, hp, m)))
    e2 = optimizar_hgb(X_tr, y_tr, X_va, y_va, cfg=cfg, semilla=1)
    assert len(intentos) == 3 and "val_mae" in intentos[0][2]
    assert e1.best_params == e2.best_params


def test_figuras_se_generan(particion):
    X_tr, y_tr, X_va, y_va = particion
    resumen = pd.DataFrame({"modelo": ["a", "b"], "val_mae": [1.0, 2.0]})
    for fig in (plots.fig_real_vs_predicho(y_va, y_va),
                plots.fig_error_por_hora(y_va, y_va, X_va["hora"]),
                plots.fig_comparacion_modelos(resumen)):
        assert fig.axes
        plt.close(fig)
