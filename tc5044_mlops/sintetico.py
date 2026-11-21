"""Datos sintéticos con la forma de MiBici, para pruebas y demostraciones.

Las pruebas automatizadas, la integración continua (CI) y la prueba de humo
del contenedor no pueden descargar los 613 MB de datos reales. Este módulo
genera datos con la misma estructura y con patrones parecidos a los reales:

- ``tabla_sintetica``: rejilla estación × hora (salida de la etapa
  ``features``) con dos tipos de estación: "de trabajo" (picos a las 8 h y
  18 h entre semana) y "de ocio" (pico al mediodía en fin de semana).
- ``viajes_sinteticos``: viajes individuales con el esquema de los CSV de
  MiBici (entrada de la etapa ``consolidar``), incluidos algunos defectos
  reales: retiros cancelados (< 1 min) y filas duplicadas.

Ambas funciones usan una semilla, así que siempre devuelven lo mismo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tc5044_mlops.features import agregar_calendario, agregar_rezagos


def _intensidad(estacion: int, horas: pd.DatetimeIndex, estaciones: int) -> np.ndarray:
    """Demanda esperada (bicis/h) de una estación en cada hora.

    La primera mitad de las estaciones es "de trabajo" y la segunda "de ocio".
    """
    hora = np.asarray(horas.hour)
    finde = np.asarray(horas.day_of_week >= 5)
    if estacion <= max(estaciones // 2, 1):   # "de trabajo"
        base = 2 + 6 * np.exp(-((hora - 8) ** 2) / 4) + 5 * np.exp(-((hora - 18) ** 2) / 4)
        base = np.where(finde, base * 0.3, base)
    else:                                     # "de ocio"
        base = 1 + 4 * np.exp(-((hora - 13) ** 2) / 9)
        base = np.where(finde, base * 2.0, base)
    return base * (0.5 + estacion / 6)


def demanda_sintetica(dias: int = 70, estaciones: int = 6, inicio: str = "2026-03-01",
                      semilla: int = 0) -> pd.DataFrame:
    """Conteos horarios por estación (columnas ``origen_id, fecha_hora, demanda``)."""
    rng = np.random.default_rng(semilla)
    horas = pd.date_range(inicio, periods=24 * dias, freq="h")
    filas = []
    for est in range(1, estaciones + 1):
        filas.append(pd.DataFrame({"origen_id": est, "fecha_hora": horas,
                                   "demanda": rng.poisson(_intensidad(est, horas, estaciones))}))
    return pd.concat(filas, ignore_index=True)


def tabla_sintetica(dias: int = 70, estaciones: int = 6, inicio: str = "2026-03-01",
                    semilla: int = 0) -> pd.DataFrame:
    """Tabla de modelado sintética: mismas columnas que ``features.tabla_modelado``.

    Pasa por las funciones de producción (``agregar_calendario`` y
    ``agregar_rezagos``), así las pruebas ejercitan el mismo código que el
    pipeline real.

    Con ``estaciones=6`` las estaciones 1-3 son "de trabajo" y 4-6 "de ocio".
    """
    dem = demanda_sintetica(dias, estaciones, inicio, semilla)
    return agregar_rezagos(agregar_calendario(dem))


def viajes_sinteticos(dias: int = 35, estaciones: int = 4, inicio: str = "2026-03-01",
                      semilla: int = 0, frac_cancelados: float = 0.03,
                      n_duplicados: int = 5) -> pd.DataFrame:
    """Viajes individuales con los encabezados de los CSV de MiBici (2025 en adelante).

    Columnas: ``Viaje_Id, Usuario_Id, Genero, Año_de_nacimiento,
    Inicio_del_viaje, Fin_del_viaje, Origen_Id, Destino_Id``.

    Se agregan dos defectos de la fuente: una fracción de retiros cancelados
    (duración < 1 min, misma estación de origen y destino) y ``n_duplicados``
    filas copiadas tal cual.
    """
    rng = np.random.default_rng(semilla)
    dem = demanda_sintetica(dias, estaciones, inicio, semilla)
    dem = dem[dem["demanda"] > 0]
    origen = np.repeat(dem["origen_id"].to_numpy(), dem["demanda"].to_numpy())
    hora_base = np.repeat(dem["fecha_hora"].to_numpy(), dem["demanda"].to_numpy())
    n = len(origen)

    inicio_viaje = (pd.to_datetime(hora_base)
                    + pd.to_timedelta(rng.integers(0, 3600, n), unit="s"))
    duracion = rng.gamma(shape=2.0, scale=7.0, size=n) + 1.5          # minutos
    destino = rng.integers(1, estaciones + 1, n)
    cancelado = rng.random(n) < frac_cancelados
    duracion[cancelado] = rng.uniform(0.05, 0.9, cancelado.sum())
    destino[cancelado] = origen[cancelado]

    viajes = pd.DataFrame({
        "Viaje_Id": np.arange(1, n + 1),
        "Usuario_Id": rng.integers(1, 5000, n),
        "Genero": rng.choice(["M", "F"], n),
        "Año_de_nacimiento": rng.integers(1960, 2008, n),
        "Inicio_del_viaje": inicio_viaje.strftime("%Y-%m-%d %H:%M:%S"),
        "Fin_del_viaje": (inicio_viaje + pd.to_timedelta(duracion, unit="m"))
        .strftime("%Y-%m-%d %H:%M:%S"),
        "Origen_Id": origen,
        "Destino_Id": destino,
    })
    if n_duplicados:
        viajes = pd.concat([viajes, viajes.sample(n_duplicados, random_state=semilla)],
                           ignore_index=True)
    return viajes.sort_values("Inicio_del_viaje", kind="mergesort").reset_index(drop=True)


def escribir_csv_mensuales(viajes: pd.DataFrame, carpeta) -> list:
    """Escribe un CSV por mes con el nombre que usa MiBici (``datos_abiertos_AAAA_MM.csv``)."""
    from pathlib import Path

    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    periodo = pd.to_datetime(viajes["Inicio_del_viaje"]).dt.strftime("%Y_%m")
    rutas = []
    for per, parte in viajes.groupby(periodo, sort=True):
        ruta = carpeta / f"datos_abiertos_{per}.csv"
        parte.to_csv(ruta, index=False)
        rutas.append(ruta)
    return rutas
