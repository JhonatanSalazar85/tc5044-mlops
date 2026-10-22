"""
Funciones de análisis exploratorio del dataset de MiBici.

Este módulo concentra la lógica de agregación y graficado para que el
notebook de EDA solo importe y llame: el notebook queda como narrativa y el
código queda versionado, probado y reutilizable (punto 5 del entregable).

Paleta: validada para daltonismo (deuteranopía/tritanopía) con separación
OKLab ΔE ≥ 9.2 en todos los pares. No se codifica información solo por color:
toda figura con más de una serie lleva leyenda.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Paleta y estilo ------------------------------------------------------
AZUL = "#2a78d6"
NARANJA = "#eb6834"
AQUA = "#1baf7a"
TINTA = "#0b0b0b"
TINTA_SEC = "#52514e"
RAMPA_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
              "#256abf", "#184f95", "#0d366b"]

DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
FIGURAS = Path("reports/figures")


def estilo() -> None:
    """Estilo sobrio y legible en impresión (el entregable va a PDF)."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#d4d3cd",
        "axes.labelcolor": TINTA_SEC,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#ebeae5",
        "grid.linewidth": 0.8,
        "xtick.color": TINTA_SEC,
        "ytick.color": TINTA_SEC,
        "text.color": TINTA,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def guardar(fig: plt.Figure, nombre: str, carpeta: Path = FIGURAS) -> Path:
    """Guarda la figura como PNG para insertarla en el reporte."""
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{nombre}.png"
    fig.savefig(ruta)
    print(f"figura -> {ruta}")
    return ruta


# --- Carga ----------------------------------------------------------------
def cargar(ruta: str | Path = "data/processed/mibici_viajes.parquet",
           solo_validos: bool = True) -> pd.DataFrame:
    """Carga el dataset limpio. Por defecto excluye los viajes marcados
    como inválidos (duración fuera de 1–1440 min)."""
    df = pd.read_parquet(ruta)
    if solo_validos and "viaje_valido" in df.columns:
        n0 = len(df)
        df = df[df["viaje_valido"]].copy()
        print(f"{len(df):,} viajes válidos de {n0:,} ({100*len(df)/n0:.2f} %)")
    return df


# --- Estadística descriptiva ---------------------------------------------
def perfil_descriptivo(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla resumen lista para pegar en el reporte."""
    filas = [
        ("Viajes", f"{len(df):,}"),
        ("Periodo", f"{df['periodo'].min()} a {df['periodo'].max()}"),
        ("Usuarios únicos", f"{df['usuario_id'].nunique():,}"),
        ("Estaciones activas",
         f"{pd.concat([df['origen_id'], df['destino_id']]).nunique():,}"),
        ("Duración mediana (min)", f"{df['duracion_min'].median():.2f}"),
        ("Duración media (min)", f"{df['duracion_min'].mean():.2f}"),
        ("Duración P95 (min)", f"{df['duracion_min'].quantile(.95):.2f}"),
        ("Viajes redondos", f"{100*df['viaje_redondo'].mean():.2f} %"),
        ("Viajes en fin de semana", f"{100*df['es_fin_semana'].mean():.2f} %"),
    ]
    if "edad" in df.columns:
        filas.append(("Edad mediana", f"{df['edad'].median():.0f}"))
    g = df["genero"].value_counts(normalize=True, dropna=False) * 100
    for k, v in g.items():
        filas.append((f"Género {k if pd.notna(k) else 'sin dato'}", f"{v:.2f} %"))
    return pd.DataFrame(filas, columns=["Indicador", "Valor"])


# --- Figuras --------------------------------------------------------------
def fig_demanda_horaria(df: pd.DataFrame) -> plt.Figure:
    """Perfil horario promedio, separando día hábil de fin de semana.

    Es la figura que justifica el problema: si el perfil tiene picos
    marcados, la demanda es predecible a partir de la hora."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    n_dias = df.groupby("es_fin_semana")["fecha"].nunique()

    for es_finde, color, etiq in [(False, AZUL, "Día hábil"),
                                  (True, NARANJA, "Fin de semana")]:
        sub = df[df["es_fin_semana"] == es_finde]
        serie = sub.groupby("hora").size() / n_dias[es_finde]
        ax.plot(serie.index, serie.values, color=color, linewidth=2,
                label=etiq, marker="o", markersize=4)

    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Viajes promedio por hora")
    ax.set_title("Perfil horario de la demanda")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc="upper left")
    return fig


def fig_heatmap_hora_dia(df: pd.DataFrame) -> plt.Figure:
    """Mapa de calor día de semana × hora: la estructura que el modelo
    debe aprender."""
    piv = (df.groupby(["dia_semana", "hora"]).size()
             .unstack(fill_value=0).reindex(index=range(7), columns=range(24))
             .fillna(0))
    dias_distintos = df.groupby("dia_semana")["fecha"].nunique()
    piv = piv.div(dias_distintos, axis=0)

    cmap = mpl.colors.LinearSegmentedColormap.from_list("azul", RAMPA_AZUL)
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    im = ax.imshow(piv.values, aspect="auto", cmap=cmap, origin="upper")
    ax.set_yticks(range(7), DIAS)
    ax.set_xticks(range(0, 24, 2), [str(h) for h in range(0, 24, 2)])
    ax.set_xlabel("Hora del día")
    ax.set_title("Demanda media por día de la semana y hora")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("Viajes por hora", color=TINTA_SEC)
    cb.outline.set_visible(False)
    return fig


def fig_distribucion_duracion(df: pd.DataFrame, corte: float = 60) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    d = df.loc[df["duracion_min"] <= corte, "duracion_min"]
    ax.hist(d, bins=60, color=AZUL, edgecolor="white", linewidth=0.5)
    med = df["duracion_min"].median()
    ax.axvline(med, color=NARANJA, linewidth=2, linestyle="--")
    ax.annotate(f"mediana {med:.1f} min", xy=(med, ax.get_ylim()[1]*0.92),
                xytext=(6, 0), textcoords="offset points",
                color=NARANJA, fontweight="bold", fontsize=9)
    ax.set_xlabel("Duración del viaje (min)")
    ax.set_ylabel("Viajes")
    ax.set_title(f"Distribución de la duración (viajes de hasta {corte:.0f} min)")
    return fig


def fig_top_estaciones(df: pd.DataFrame, n: int = 15) -> plt.Figure:
    """Concentración de la demanda: cuántas estaciones explican el grueso."""
    conteo = df["origen_id"].value_counts().head(n).sort_values()
    etiquetas = _etiquetas_estacion(df, conteo.index)

    fig, ax = plt.subplots(figsize=(7.2, 0.26 * n + 1.2))
    ax.barh(range(len(conteo)), conteo.values, color=AZUL, height=0.72)
    ax.set_yticks(range(len(conteo)), etiquetas, fontsize=8)
    ax.set_xlabel("Viajes originados")
    ax.set_title(f"Las {n} estaciones con más salidas")
    for i, v in enumerate(conteo.values):
        ax.annotate(f"{v/1000:.0f}k", xy=(v, i), xytext=(4, 0),
                    textcoords="offset points", va="center",
                    fontsize=8, color=TINTA_SEC)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.08)
    return fig


def _etiquetas_estacion(df: pd.DataFrame, ids) -> list[str]:
    if "origen_estacion_nombre" in df.columns:
        mapa = (df.dropna(subset=["origen_estacion_nombre"])
                  .groupby("origen_id")["origen_estacion_nombre"].first())
        return [str(mapa.get(i, f"Est. {i}"))[:34] for i in ids]
    return [f"Estación {i}" for i in ids]


def fig_serie_mensual(df: pd.DataFrame) -> plt.Figure:
    """Volumen mensual: la línea base contra la cual se medirá el drift."""
    serie = df.groupby("periodo").size()
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.plot(range(len(serie)), serie.values / 1000, color=AZUL,
            linewidth=2, marker="o", markersize=5)
    ax.set_xticks(range(len(serie)), serie.index, rotation=60, fontsize=8)
    ax.set_ylabel("Viajes (miles)")
    ax.set_title("Volumen mensual de viajes")
    media = serie.mean() / 1000
    ax.axhline(media, color=TINTA_SEC, linewidth=1, linestyle=":")
    ax.annotate(f"media {media:.0f}k", xy=(0, media), xytext=(2, 5),
                textcoords="offset points", fontsize=8, color=TINTA_SEC)
    ax.margins(x=0.02)
    return fig


def fig_genero_edad(df: pd.DataFrame) -> plt.Figure:
    """Composición demográfica: material para la discusión de equidad."""
    sub = df.dropna(subset=["edad"])
    sub = sub[sub["genero"].isin(["M", "F"])]
    bins = np.arange(15, 76, 5)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for gen, color, etiq in [("M", AZUL, "Masculino"), ("F", NARANJA, "Femenino")]:
        v = sub.loc[sub["genero"] == gen, "edad"]
        ax.hist(v, bins=bins, histtype="step", linewidth=2,
                color=color, label=f"{etiq} ({100*len(v)/len(sub):.0f} %)")
    ax.set_xlabel("Edad (años)")
    ax.set_ylabel("Viajes")
    ax.set_title("Distribución de edad por género")
    ax.legend()
    return fig


def generar_todas(df: pd.DataFrame, carpeta: Path = FIGURAS) -> list[Path]:
    """Genera y guarda el set completo de figuras del EDA."""
    estilo()
    salidas = []
    for nombre, func in [
        ("eda_perfil_horario", fig_demanda_horaria),
        ("eda_heatmap_dia_hora", fig_heatmap_hora_dia),
        ("eda_duracion", fig_distribucion_duracion),
        ("eda_top_estaciones", fig_top_estaciones),
        ("eda_serie_mensual", fig_serie_mensual),
        ("eda_genero_edad", fig_genero_edad),
    ]:
        fig = func(df)
        salidas.append(guardar(fig, nombre, carpeta))
        plt.close(fig)
    return salidas
