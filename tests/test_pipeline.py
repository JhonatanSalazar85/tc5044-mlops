"""Pruebas del pipeline de preprocesamiento (escalado, TargetEncoder, PCA)."""

import numpy as np
from sklearn.linear_model import Ridge

from tc5044_mlops.config import cargar_params
from tc5044_mlops.features import particion_temporal
from tc5044_mlops.pipeline import (
    PerfilHorarioPCA,
    agrupar_estaciones,
    columnas_entrada,
    construir_pipeline,
    construir_preprocesador,
)


def _split(tabla, cortes):
    tr, va, _ = particion_temporal(tabla, **cortes)
    cols = columnas_entrada()
    return tr[cols], tr["demanda"], va[cols], va["demanda"]


def test_preprocesador_produce_matriz_sin_nulos(tabla_sintetica, cortes_sinteticos):
    X_tr, y_tr, X_va, _ = _split(tabla_sintetica, cortes_sinteticos)
    pre = construir_preprocesador().fit(X_tr, y_tr)
    Z = pre.transform(X_va)
    cfg = cargar_params("pipeline")
    esperado = len(cfg["numericas"]) + 1 + cfg["pca_componentes"]
    assert Z.shape == (len(X_va), esperado)
    assert not np.isnan(Z).any(), "la imputación debe cubrir los rezagos de la primera semana"


def test_nombres_de_variables_salen_del_preprocesador(tabla_sintetica, cortes_sinteticos):
    X_tr, y_tr, _, _ = _split(tabla_sintetica, cortes_sinteticos)
    nombres = construir_preprocesador().fit(X_tr, y_tr).get_feature_names_out()
    assert "estacion_te__origen_id" in nombres
    assert "perfil_pca__pca_perfil_1" in nombres


def test_perfil_pca_separa_tipos_de_uso(tabla_sintetica):
    X = tabla_sintetica[["origen_id", "hora", "es_fin_semana"]]
    pca = PerfilHorarioPCA(n_componentes=2).fit(X, tabla_sintetica["demanda"])
    grupos = agrupar_estaciones(pca, n_clusters=2)["grupo_uso"]
    # las estaciones 1-3 (trabajo) y 4-6 (ocio) deben quedar en grupos distintos
    assert grupos.loc[[1, 2, 3]].nunique() == 1
    assert grupos.loc[[4, 5, 6]].nunique() == 1
    assert grupos.loc[1] != grupos.loc[4]
    assert pca.varianza_explicada_[0] > 0.5


def test_estacion_nueva_no_rompe_la_transformacion(tabla_sintetica):
    X = tabla_sintetica[["origen_id", "hora", "es_fin_semana"]]
    pca = PerfilHorarioPCA(n_componentes=2).fit(X, tabla_sintetica["demanda"])
    nueva = X.head(3).assign(origen_id=999)
    assert np.allclose(pca.transform(nueva), 0.0)


def test_pipeline_completo_entrena_y_predice(tabla_sintetica, cortes_sinteticos):
    X_tr, y_tr, X_va, y_va = _split(tabla_sintetica, cortes_sinteticos)
    pipe = construir_pipeline(Ridge(alpha=1.0)).fit(X_tr, y_tr)
    pred = pipe.predict(X_va)
    assert pred.shape == y_va.shape
    # mejor que predecir siempre la media del train
    mae_modelo = np.abs(pred - y_va).mean()
    mae_media = np.abs(y_tr.mean() - y_va).mean()
    assert mae_modelo < mae_media


def test_el_preprocesador_solo_aprende_del_train(tabla_sintetica, cortes_sinteticos):
    X_tr, y_tr, X_va, _ = _split(tabla_sintetica, cortes_sinteticos)
    pre = construir_preprocesador().fit(X_tr, y_tr)
    escalador = pre.named_transformers_["numericas"].named_steps["escalar"]
    assert np.allclose(escalador.mean_[0], X_tr["hora_sin"].mean(), atol=1e-6)
