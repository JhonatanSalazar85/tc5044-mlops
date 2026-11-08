"""Pipeline de preprocesamiento y modelado con scikit-learn.

Un solo objeto `Pipeline` encadena el preprocesamiento y el estimador. Esto
garantiza que las transformaciones se ajusten **solo con el conjunto de
entrenamiento** y se apliquen igual en validación, prueba y producción (sin
fuga de información ni diferencias entre entrenamiento y servicio).

Etapas del preprocesador (`ColumnTransformer`):

1. ``numericas``  -> imputación por mediana + ``StandardScaler``.
   Los rezagos tienen nulos en la primera semana de cada estación. El escalado
   es indispensable para los modelos lineales (Ridge, Poisson) y no afecta a
   los árboles.
2. ``estacion_te`` -> ``TargetEncoder`` + escalado.
   Reemplaza el id de la estación (cientos de categorías) por su demanda
   media suavizada. En `fit_transform` usa *cross fitting* (k pliegues), así
   la codificación de una fila nunca usa su propio valor objetivo.
3. ``perfil_pca`` -> ``PerfilHorarioPCA`` + escalado.
   Construye el perfil horario de cada estación (proporción de su demanda en
   cada hora, laborable y fin de semana = 48 valores) y lo reduce con PCA. Las
   componentes describen el *tipo de uso* (estaciones de ida al trabajo,
   de regreso, de ocio de fin de semana) independientemente del volumen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.utils.validation import check_is_fitted

from tc5044_mlops.config import cargar_params


class PerfilHorarioPCA(TransformerMixin, BaseEstimator):
    """Componentes principales del perfil horario de cada estación.

    Requiere el objetivo ``y`` en `fit` (la demanda), por eso solo debe
    ajustarse con el conjunto de entrenamiento; dentro de un `Pipeline` eso
    ocurre automáticamente.

    Parameters
    ----------
    n_componentes
        Número de componentes principales a conservar.
    col_estacion, col_hora, col_fin_semana
        Columnas de entrada (en ese orden).
    random_state
        Semilla del PCA (solo relevante con el solver aleatorizado).
    """

    def __init__(self, n_componentes: int = 3, col_estacion: str = "origen_id",
                 col_hora: str = "hora", col_fin_semana: str = "es_fin_semana",
                 random_state: int | None = None):
        self.n_componentes = n_componentes
        self.col_estacion = col_estacion
        self.col_hora = col_hora
        self.col_fin_semana = col_fin_semana
        self.random_state = random_state

    def _como_df(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X[[self.col_estacion, self.col_hora, self.col_fin_semana]]
        return pd.DataFrame(np.asarray(X)[:, :3],
                            columns=[self.col_estacion, self.col_hora, self.col_fin_semana])

    def perfiles(self, X, y) -> pd.DataFrame:
        """Matriz estación × (fin_semana, hora) con la proporción de demanda."""
        d = self._como_df(X).copy()
        d["_y"] = np.asarray(y, dtype=float)
        tabla = d.pivot_table(index=self.col_estacion,
                              columns=[self.col_fin_semana, self.col_hora],
                              values="_y", aggfunc="mean")
        columnas = pd.MultiIndex.from_product([[0, 1], range(24)])
        tabla = tabla.reindex(columns=columnas).fillna(0.0)
        total = tabla.sum(axis=1).replace(0, 1.0)
        return tabla.div(total, axis=0)

    def fit(self, X, y):
        if y is None:
            raise ValueError("PerfilHorarioPCA necesita el objetivo y para construir los perfiles")
        perfiles = self.perfiles(X, y)
        n = int(min(self.n_componentes, *perfiles.shape))
        self.pca_ = PCA(n_components=n, random_state=self.random_state).fit(perfiles.to_numpy())
        # Se guardan arreglos de numpy (no DataFrames) para que el modelo se pueda
        # serializar con skops, el formato seguro que MLflow 3 usa por defecto.
        self.nombres_ = np.asarray([f"pca_perfil_{i + 1}" for i in range(n)], dtype=object)
        self.estaciones_ = np.asarray(perfiles.index, dtype=float)       # ordenadas
        self.componentes_ = self.pca_.transform(perfiles.to_numpy())
        self.perfiles_ = perfiles.to_numpy()
        self.varianza_explicada_ = self.pca_.explained_variance_ratio_
        self.n_features_in_ = 3
        return self

    def transform(self, X):
        check_is_fitted(self, "componentes_")
        est = np.asarray(self._como_df(X)[self.col_estacion], dtype=float)
        pos = np.clip(np.searchsorted(self.estaciones_, est), 0, len(self.estaciones_) - 1)
        conocida = self.estaciones_[pos] == est
        # Una estación nueva (no vista en train) cae en el centro del espacio PCA (0)
        salida = np.zeros((len(est), self.componentes_.shape[1]))
        salida[conocida] = self.componentes_[pos[conocida]]
        return salida

    def tabla_componentes(self) -> pd.DataFrame:
        """Componentes PCA por estación (para análisis y figuras)."""
        check_is_fitted(self, "componentes_")
        return pd.DataFrame(self.componentes_, columns=list(self.nombres_),
                            index=pd.Index(self.estaciones_.astype(int), name=self.col_estacion))

    def tabla_perfiles(self) -> pd.DataFrame:
        """Perfil horario normalizado por estación (48 columnas: laborable | fin de semana)."""
        check_is_fitted(self, "perfiles_")
        return pd.DataFrame(self.perfiles_,
                            index=pd.Index(self.estaciones_.astype(int), name=self.col_estacion))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "nombres_")
        return self.nombres_


def columnas_entrada(cfg: dict | None = None) -> list[str]:
    """Columnas que el pipeline recibe (en orden, sin duplicados)."""
    cfg = cfg or cargar_params("pipeline")
    cols = list(cfg["numericas"]) + [cfg["col_estacion"], cfg["col_hora"], cfg["col_fin_semana"]]
    return list(dict.fromkeys(cols))


def construir_preprocesador(cfg: dict | None = None, semilla: int | None = None
                            ) -> ColumnTransformer:
    """`ColumnTransformer` con las tres ramas descritas en el docstring del módulo."""
    cfg = cfg or cargar_params("pipeline")
    semilla = cargar_params("semilla") if semilla is None else semilla

    numericas = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    estacion = Pipeline([
        ("target_encoder", TargetEncoder(target_type="continuous", smooth="auto",
                                         cv=cfg["target_encoder_cv"], shuffle=True,
                                         random_state=semilla)),
        ("escalar", StandardScaler()),
    ])
    perfil = Pipeline([
        ("pca", PerfilHorarioPCA(n_componentes=cfg["pca_componentes"],
                                 col_estacion=cfg["col_estacion"], col_hora=cfg["col_hora"],
                                 col_fin_semana=cfg["col_fin_semana"], random_state=semilla)),
        ("escalar", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("numericas", numericas, list(cfg["numericas"])),
            ("estacion_te", estacion, [cfg["col_estacion"]]),
            ("perfil_pca", perfil, [cfg["col_estacion"], cfg["col_hora"], cfg["col_fin_semana"]]),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def construir_pipeline(estimador, cfg: dict | None = None, semilla: int | None = None,
                       preprocesar: bool = True) -> Pipeline:
    """Pipeline completo ``preprocesador -> modelo``.

    Con ``preprocesar=False`` se devuelve ``Pipeline([("modelo", estimador)])``:
    se usa para la línea base, que lee una columna cruda (el rezago semanal).
    """
    if not preprocesar:
        return Pipeline([("modelo", estimador)])
    return Pipeline([
        ("preprocesador", construir_preprocesador(cfg, semilla)),
        ("modelo", estimador),
    ])


def agrupar_estaciones(perfil: PerfilHorarioPCA, n_clusters: int, semilla: int = 42
                       ) -> pd.DataFrame:
    """Agrupa las estaciones por tipo de uso (KMeans sobre las componentes PCA).

    Es un análisis descriptivo para el reporte; el modelo usa las componentes
    continuas, no la etiqueta del grupo.
    """
    comp = perfil.tabla_componentes()
    k = int(min(n_clusters, len(comp)))
    etiquetas = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit_predict(comp)
    out = comp.copy()
    out["grupo_uso"] = etiquetas
    return out
