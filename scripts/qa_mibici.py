#!/usr/bin/env python3
"""
Control de calidad del dataset consolidado de MiBici.

Revisa lo que el resumen de `consolidar` no muestra: duplicados, duraciones
imposibles, edades absurdas, cobertura temporal con huecos y estaciones que
aparecen en los viajes pero no en la nomenclatura.

Uso:
    python scripts/qa_mibici.py
    python scripts/qa_mibici.py --parquet data/interim/mibici_viajes.parquet \
                                --nomenclatura data/raw/mibici/nomenclatura
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def _pct(n: int, total: int) -> str:
    return f"{n:,} ({100*n/total:.3f} %)" if total else f"{n:,}"


def _seccion(titulo: str) -> None:
    print(f"\n{'─'*68}\n{titulo}\n{'─'*68}")


def cargar_nomenclatura(ruta: Path) -> pd.DataFrame | None:
    """Lee el CSV de estaciones y detecta las columnas id / lat / lon."""
    archivos = sorted(ruta.glob("*.csv")) if ruta.is_dir() else [ruta]
    if not archivos:
        return None

    df = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(archivos[-1], encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        return None

    cols = {_slug(c): c for c in df.columns}
    def buscar(*alias):
        for a in alias:
            if a in cols:
                return cols[a]
        return None

    c_id = buscar("id", "station_id", "estacion_id", "id_estacion")
    c_lat = buscar("latitude", "latitud", "lat")
    c_lon = buscar("longitude", "longitud", "lon", "lng")
    if not c_id:
        print(f"  [aviso] no reconocí la columna de id. Columnas: {list(df.columns)}")
        return None

    out = df.rename(columns={c_id: "estacion_id"})
    if c_lat and c_lon:
        out = out.rename(columns={c_lat: "latitud", c_lon: "longitud"})
    out["estacion_id"] = pd.to_numeric(out["estacion_id"], errors="coerce").astype("Int64")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default="data/interim/mibici_viajes.parquet")
    ap.add_argument("--nomenclatura", default="data/raw/mibici/nomenclatura")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    n = len(df)
    print(f"{n:,} viajes  |  {df['periodo'].min()} a {df['periodo'].max()}")

    # ------------------------------------------------------------------
    _seccion("1. Duplicados")
    dup_id = int(df["viaje_id"].duplicated().sum())
    dup_full = int(df.duplicated(subset=["usuario_id", "inicio_viaje",
                                         "origen_id", "destino_id"]).sum())
    print(f"viaje_id repetido        : {_pct(dup_id, n)}")
    print(f"viaje lógico repetido    : {_pct(dup_full, n)}")
    if dup_id:
        print("  -> revisa si algún mes quedó descargado dos veces "
              "(p. ej. *-1.csv) o si MiBici republicó un periodo")

    # ------------------------------------------------------------------
    _seccion("2. Duración del viaje")
    d = df["duracion_min"]
    neg = int((d < 0).sum())
    cero = int((d == 0).sum())
    corto = int(((d > 0) & (d < 1)).sum())
    largo = int((d > 180).sum())
    dia = int((d > 1440).sum())
    print(f"negativa                 : {_pct(neg, n)}")
    print(f"exactamente 0 min        : {_pct(cero, n)}")
    print(f"menor a 1 min            : {_pct(corto, n)}")
    print(f"mayor a 3 h              : {_pct(largo, n)}")
    print(f"mayor a 24 h             : {_pct(dia, n)}")
    print("\npercentiles (min):")
    print(d.describe(percentiles=[.01, .25, .5, .75, .95, .99, .999]).to_string())

    # ------------------------------------------------------------------
    _seccion("3. Edad del usuario")
    anio_max = int(df["inicio_viaje"].dt.year.max())
    edad = anio_max - df["anio_nacimiento"]
    print(f"año de nacimiento < 1930 : {_pct(int((df['anio_nacimiento'] < 1930).sum()), n)}")
    print(f"edad < 12 años           : {_pct(int((edad < 12).sum()), n)}")
    print(f"edad > 90 años           : {_pct(int((edad > 90).sum()), n)}")
    print(f"nulo                     : {_pct(int(df['anio_nacimiento'].isna().sum()), n)}")

    _seccion("4. Género")
    print(df["genero"].value_counts(dropna=False).to_string())

    # ------------------------------------------------------------------
    _seccion("5. Cobertura temporal")
    por_mes = df.groupby("periodo").size()
    esperados = pd.period_range(df["periodo"].min(), df["periodo"].max(), freq="M")
    faltan = [str(p) for p in esperados if str(p) not in por_mes.index]
    print(por_mes.to_string())
    print(f"\nmeses faltantes: {faltan if faltan else 'ninguno'}")
    var = por_mes.pct_change().abs()
    saltos = var[var > 0.35]
    if len(saltos):
        print("\nsaltos de volumen > 35 % respecto al mes previo "
              "(candidatos a drift o a mes incompleto):")
        print(saltos.round(3).to_string())

    # ------------------------------------------------------------------
    _seccion("6. Estaciones")
    nom = cargar_nomenclatura(Path(args.nomenclatura))
    usadas = pd.unique(pd.concat([df["origen_id"], df["destino_id"]]).dropna())
    print(f"estaciones distintas en viajes : {len(usadas):,}")
    if nom is not None:
        catalogo = set(nom["estacion_id"].dropna().astype(int))
        print(f"estaciones en nomenclatura     : {len(catalogo):,}")
        huerfanas = sorted(set(int(x) for x in usadas) - catalogo)
        sin_uso = sorted(catalogo - set(int(x) for x in usadas))
        print(f"en viajes pero NO en catálogo  : {len(huerfanas)} {huerfanas[:15]}")
        print(f"en catálogo pero sin viajes    : {len(sin_uso)} {sin_uso[:15]}")
        if huerfanas:
            print("  -> esos viajes se quedarán sin lat/lon al hacer el join")
        if {"latitud", "longitud"}.issubset(nom.columns):
            print(f"\ncatálogo sin coordenadas       : "
                  f"{int(nom[['latitud','longitud']].isna().any(axis=1).sum())}")
    else:
        print("  (no se pudo leer la nomenclatura; revisa la ruta)")

    mismo = int((df["origen_id"] == df["destino_id"]).sum())
    print(f"\nviajes con origen == destino   : {_pct(mismo, n)}")

    _seccion("Resumen")
    problemas = []
    if dup_id: problemas.append(f"{dup_id:,} viaje_id duplicados")
    if neg: problemas.append(f"{neg:,} duraciones negativas")
    if dia: problemas.append(f"{dia:,} viajes de más de 24 h")
    if faltan: problemas.append(f"meses faltantes: {faltan}")
    print("\n".join(f"  - {p}" for p in problemas) if problemas
          else "  sin hallazgos bloqueantes")


if __name__ == "__main__":
    main()
