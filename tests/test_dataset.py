"""Pruebas de consolidación y limpieza de extremo a extremo con datos mínimos.

No descargan nada: escriben dos CSV con encabezados distintos (como los ha
publicado MiBici a lo largo de los años) y verifican el esquema resultante.
"""

import json

import pandas as pd

from tc5044_mlops.dataset import consolidar_dataset, slug
from tc5044_mlops.limpieza import limpiar


def _csv_mes(ruta, encabezados, filas):
    pd.DataFrame(filas, columns=encabezados).to_csv(ruta, index=False)


def test_slug_normaliza_acentos_y_espacios():
    assert slug("Año de nacimiento ") == "ano_de_nacimiento"
    assert slug("Inicio_del_viaje") == "inicio_del_viaje"


def test_consolidar_unifica_encabezados_de_distintos_anios(tmp_path):
    viajes = tmp_path / "viajes"
    viajes.mkdir()
    _csv_mes(viajes / "datos_abiertos_2025_01.csv",
             ["Viaje_Id", "Usuario_Id", "Genero", "Año_de_nacimiento",
              "Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"],
             [[1, 7, "M", 1990, "2025-01-01 08:00:00", "2025-01-01 08:12:00", 10, 11]])
    _csv_mes(viajes / "datos_abiertos_2025_02.csv",
             ["Trip_Id", "User_Id", "Gender", "Year_of_Birth",
              "Trip_Start", "Trip_End", "Origin_Id", "Destination_Id"],
             [[2, 8, "f", 1985, "2025-02-01 09:00:00", "2025-02-01 09:30:00", 11, 10]])
    df = consolidar_dataset(tmp_path, tmp_path / "viajes.parquet")
    assert list(df["viaje_id"]) == [1, 2]
    assert list(df["periodo"]) == ["2025-01", "2025-02"]
    assert list(df["genero"]) == ["M", "F"]
    assert df["duracion_min"].tolist() == [12.0, 30.0]


def test_limpiar_marca_sin_borrar_y_genera_reporte(tmp_path):
    entrada = tmp_path / "interim.parquet"
    pd.DataFrame({
        "viaje_id": pd.array([1, 2, 3, 3], dtype="Int64"),
        "usuario_id": pd.array([1, 2, 3, 3], dtype="Int64"),
        "genero": ["M", "F", "M", "M"],
        "anio_nacimiento": pd.array([1990, 1800, 2000, 2000], dtype="Int64"),
        "inicio_viaje": pd.to_datetime(["2025-01-01 08:00"] * 4),
        "fin_viaje": pd.to_datetime(["2025-01-01 08:10:00", "2025-01-01 08:00:30",
                                     "2025-01-01 09:00:00", "2025-01-01 09:00:00"]),
        "origen_id": pd.array([10, 10, 11, 11], dtype="Int64"),
        "destino_id": pd.array([11, 10, 10, 10], dtype="Int64"),
        "periodo": ["2025-01"] * 4,
        "archivo_origen": ["a.csv"] * 4,
        "duracion_min": [10.0, 0.5, 60.0, 60.0],
    }).to_parquet(entrada)
    par = {"duracion_min_valida": 1.0, "duracion_max_valida": 1440.0,
           "edad_min_valida": 12, "edad_max_valida": 90}
    rep = limpiar(entrada, tmp_path / "sin_nomenclatura", tmp_path / "salida.parquet",
                  tmp_path / "limpieza.json", par)
    salida = pd.read_parquet(tmp_path / "salida.parquet")
    assert rep["duplicados_eliminados"] == 1          # la fila 4 es copia exacta
    assert len(salida) == 3                           # el viaje corto se marca, no se borra
    assert salida["viaje_valido"].tolist() == [True, False, True]
    assert rep["edades_anuladas"] == 1                # nacido en 1800
    assert json.loads((tmp_path / "limpieza.json").read_text())["viajes_salida"] == 3
