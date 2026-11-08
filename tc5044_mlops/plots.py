"""Figuras de evaluación de modelos (se guardan en reports/figures y en MLflow).

Cada función devuelve una figura de matplotlib; el llamador decide si la
guarda en disco o la registra como artefacto con ``mlflow.log_figure``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # sin pantalla: funciona en WSL, CI y servidores

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tc5044_mlops.modeling.evaluate import error_por_hora  # noqa: E402


def fig_real_vs_predicho(y_real, y_pred, titulo: str = "", muestra: int = 20_000,
                         semilla: int = 42):
    """Dispersión real vs. predicho (muestra aleatoria para no saturar)."""
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    if len(y_real) > muestra:
        idx = np.random.default_rng(semilla).choice(len(y_real), muestra, replace=False)
        y_real, y_pred = y_real[idx], y_pred[idx]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_real, y_pred, s=4, alpha=0.25)
    tope = max(y_real.max(), y_pred.max(), 1)
    ax.plot([0, tope], [0, tope], color="black", lw=1, ls="--", label="predicción perfecta")
    ax.set(xlabel="demanda real (bicis/h)", ylabel="demanda predicha (bicis/h)",
           title=titulo or "Real vs. predicho")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def fig_error_por_hora(y_real, y_pred, horas, titulo: str = ""):
    """MAE por hora del día frente a la demanda media de esa hora."""
    tabla = error_por_hora(y_real, y_pred, horas)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(tabla.index, tabla["demanda_media"], color="#c9d6e3", label="demanda media")
    ax.plot(tabla.index, tabla["mae"], marker="o", color="#b5402a", label="MAE")
    ax.set(xlabel="hora del día", ylabel="bicis/h", xticks=range(0, 24, 2),
           title=titulo or "Error por hora del día")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_serie_estacion(fechas, y_real, y_pred, estacion, horas: int = 24 * 7):
    """Serie real vs. predicha de una estación durante la primera semana de prueba."""
    d = pd.DataFrame({"fecha_hora": pd.to_datetime(fechas), "real": np.asarray(y_real),
                      "pred": np.clip(np.asarray(y_pred, dtype=float), 0, None)})
    d = d.sort_values("fecha_hora").head(horas)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(d["fecha_hora"], d["real"], lw=1.2, label="real")
    ax.plot(d["fecha_hora"], d["pred"], lw=1.2, label="predicho")
    ax.set(ylabel="bicis/h", title=f"Estación {estacion}: primera semana del conjunto de prueba")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_comparacion_modelos(resumen: pd.DataFrame, metrica: str = "val_mae"):
    """Barras con la métrica de selección de cada modelo candidato."""
    r = resumen.sort_values(metrica)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    colores = ["#2a7ab5" if i == 0 else "#9bb7cf" for i in range(len(r))]
    ax.barh(r["modelo"], r[metrica], color=colores)
    for y, v in enumerate(r[metrica]):
        ax.text(v, y, f" {v:.3f}", va="center")
    ax.invert_yaxis()
    ax.set_xlim(0, r[metrica].max() * 1.15)
    ax.set(xlabel=metrica, title="Comparación de modelos (menor es mejor)")
    fig.tight_layout()
    return fig


def fig_perfiles_pca(grupos: pd.DataFrame, perfiles: pd.DataFrame, varianza):
    """Estaciones en el plano PCA del perfil horario y perfil medio de cada grupo."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for g, sub in grupos.groupby("grupo_uso"):
        ax1.scatter(sub["pca_perfil_1"], sub["pca_perfil_2"] if "pca_perfil_2" in sub else 0,
                    s=30, label=f"grupo {g} ({len(sub)})")
        medio = perfiles.loc[sub.index].mean()
        ax2.plot(range(48), medio.to_numpy(), label=f"grupo {g}")
    ax1.set(xlabel=f"PC1 ({100 * varianza[0]:.0f} %)",
            ylabel=f"PC2 ({100 * varianza[1]:.0f} %)" if len(varianza) > 1 else "",
            title="Estaciones según su perfil horario")
    ax1.legend(fontsize=8)
    ax2.axvline(23.5, color="grey", ls=":")
    ax2.set_xticks([0, 6, 12, 18, 24, 30, 36, 42],
                   ["0", "6", "12", "18", "0", "6", "12", "18"])
    ax2.set(xlabel="hora (laborable | fin de semana)", ylabel="proporción de la demanda",
            title="Perfil horario medio por grupo")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    return fig
