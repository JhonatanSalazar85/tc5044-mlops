#!/usr/bin/env python3
"""
Etapa de limpieza y enriquecimiento del dataset de MiBici.

Entrada : data/interim/mibici_viajes.parquet  (unión fiel de los CSV)
Salida  : data/processed/mibici_viajes.parquet (banderas + features)
          reports/limpieza.json                (métricas trazables)

Criterio de diseño: NO se borran filas salvo duplicados exactos (defecto de
la fuente). Todo lo demás se MARCA con banderas booleanas, de modo que el
conteo original siga disponible y cada decisión de filtrado sea explícita y
auditable aguas abajo.

Uso:
    python scripts/limpiar_mibici.py
    python scripts/limpiar_mibici.py --params params.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULTS = {
    "duracion_min_valida": 1.0,      # minutos; por debajo = retiro cancelado
    "duracion_max_valida": 1440.0,   # minutos; por encima = bici no devuelta
    "edad_min_valida": 12,
    "edad_max_valida": 90,
}


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def cargar_params(ruta: Path | None) -> dict:
    p = dict(DEFAULTS)
    if ruta and ruta.exists():
        try:
            import yaml
            cfg = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
            p.update((cfg.get("limpiar") or {}))
        except ImportError:
            print("[aviso] pyyaml no instalado; uso los valores por defecto")
    return p


def cargar_nomenclatura(ruta: Path) -> pd.DataFrame | None:
    """Catálogo de estaciones. Ojo: es una foto del estado ACTUAL del sistema,
    no un histórico, así que estaciones retiradas no aparecen."""
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
        return next((cols[a] for a in alias if a in cols), None)

    c_id = buscar("id", "station_id", "estacion_id", "id_estacion")
    if not c_id:
        print(f"[aviso] no reconocí el id en la nomenclatura: {list(df.columns)}")
        return None
    ren = {c_id: "estacion_id"}
    if (c := buscar("latitude", "latitud", "lat")):
        ren[c] = "latitud"
    if (c := buscar("longitude", "longitud", "lon", "lng")):
        ren[c] = "longitud"
    if (c := buscar("name", "nombre", "location", "ubicacion")):
        ren[c] = "estacion_nombre"

    out = df.rename(columns=ren)
    out["estacion_id"] = pd.to_numeric(out["estacion_id"], errors="coerce").astype("Int64")
    keep = [c for c in ("estacion_id", "estacion_nombre", "latitud", "longitud")
            if c in out.columns]
    return out[keep].dropna(subset=["estacion_id"]).drop_duplicates("estacion_id")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", default="data/interim/mibici_viajes.parquet")
    ap.add_argument("--nomenclatura", default="data/raw/mibici/nomenclatura")
    ap.add_argument("--salida", default="data/processed/mibici_viajes.parquet")
    ap.add_argument("--reporte", default="reports/limpieza.json")
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()

    par = cargar_params(Path(args.params))
    df = pd.read_parquet(args.entrada)
    n0 = len(df)
    rep: dict = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entrada": args.entrada,
        "params": par,
        "viajes_entrada": n0,
    }
    print(f"entrada: {n0:,} viajes ({df['periodo'].min()} a {df['periodo'].max()})")

    # -- 1. duplicados exactos: único caso en que sí se borra ---------------
    antes = len(df)
    df = df.drop_duplicates(subset=["viaje_id"], keep="first").reset_index(drop=True)
    rep["duplicados_eliminados"] = antes - len(df)
    print(f"duplicados exactos eliminados : {rep['duplicados_eliminados']:,}")

    # -- 2. bandera de validez por duración ---------------------------------
    dmin, dmax = par["duracion_min_valida"], par["duracion_max_valida"]
    corto = df["duracion_min"] < dmin
    largo = df["duracion_min"] > dmax
    df["viaje_valido"] = ~(corto | largo)
    rep["marcados_duracion_corta"] = int(corto.sum())
    rep["marcados_duracion_larga"] = int(largo.sum())
    rep["viajes_validos"] = int(df["viaje_valido"].sum())
    print(f"marcados duración < {dmin} min   : {rep['marcados_duracion_corta']:,}")
    print(f"marcados duración > {dmax} min: {rep['marcados_duracion_larga']:,}")

    # -- 3. edad: valores imposibles a nulo ---------------------------------
    anio_ref = df["inicio_viaje"].dt.year
    edad = anio_ref - df["anio_nacimiento"]
    fuera = (edad < par["edad_min_valida"]) | (edad > par["edad_max_valida"])
    rep["edades_anuladas"] = int(fuera.fillna(False).sum())
    df.loc[fuera.fillna(False), "anio_nacimiento"] = pd.NA
    df["edad"] = (anio_ref - df["anio_nacimiento"]).astype("Int64")
    print(f"edades imposibles anuladas    : {rep['edades_anuladas']:,}")

    # -- 4. join con el catálogo de estaciones ------------------------------
    nom = cargar_nomenclatura(Path(args.nomenclatura))
    if nom is not None:
        for lado in ("origen", "destino"):
            sub = nom.add_prefix(f"{lado}_").rename(
                columns={f"{lado}_estacion_id": f"{lado}_id"})
            df = df.merge(sub, on=f"{lado}_id", how="left")
        falta_o = df["origen_latitud"].isna() if "origen_latitud" in df else pd.Series(False, index=df.index)
        falta_d = df["destino_latitud"].isna() if "destino_latitud" in df else pd.Series(False, index=df.index)
        df["estacion_catalogada"] = ~(falta_o | falta_d)
        rep["viajes_sin_catalogo"] = int((~df["estacion_catalogada"]).sum())
        print(f"viajes sin estación en catálogo: {rep['viajes_sin_catalogo']:,}")
    else:
        df["estacion_catalogada"] = pd.NA
        rep["viajes_sin_catalogo"] = None
        print("[aviso] sin nomenclatura: no se agregaron coordenadas")

    # -- 5. features temporales ---------------------------------------------
    ini = df["inicio_viaje"]
    df["hora"] = ini.dt.hour.astype("Int8")
    df["dia_semana"] = ini.dt.dayofweek.astype("Int8")   # 0=lunes
    df["es_fin_semana"] = df["dia_semana"].isin([5, 6])
    df["mes"] = ini.dt.month.astype("Int8")
    df["fecha"] = ini.dt.date
    df["viaje_redondo"] = df["origen_id"] == df["destino_id"]
    rep["viajes_redondos"] = int(df["viaje_redondo"].sum())

    # -- 6. salida -----------------------------------------------------------
    sal = Path(args.salida)
    sal.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(sal, index=False, compression="zstd")

    rep["viajes_salida"] = len(df)
    rep["pct_validos"] = round(100 * rep["viajes_validos"] / len(df), 4)
    rep["salida"] = str(sal)
    rep["mb_salida"] = round(sal.stat().st_size / 1048576, 1)

    rpt = Path(args.reporte)
    rpt.parent.mkdir(parents=True, exist_ok=True)
    rpt.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(df):,} viajes -> {sal} ({rep['mb_salida']} MB)")
    print(f"válidos para modelar: {rep['viajes_validos']:,} ({rep['pct_validos']} %)")
    print(f"reporte: {rpt}")


if __name__ == "__main__":
    main()
