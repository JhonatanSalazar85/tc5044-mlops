"""
Etapa de limpieza y enriquecimiento del dataset de MiBici.

Entrada : data/interim/mibici_viajes.parquet  (unión fiel de los CSV)
Salida  : data/processed/mibici_viajes.parquet (banderas + features)
          reports/limpieza.json                (métricas trazables)

Criterio de diseño: NO se borran filas salvo duplicados exactos (defecto de
la fuente). Todo lo demás se MARCA con banderas booleanas, de modo que el
conteo original siga disponible y cada decisión de filtrado sea explícita y
auditable aguas abajo. Los umbrales se leen de la sección `limpiar` de
params.yaml.

Uso (desde la raíz del proyecto):
    python -m tc5044_mlops.limpieza

Antes vivía en scripts/limpiar_mibici.py (se movió al paquete en la Fase 2).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from tc5044_mlops.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    cargar_params,
)
from tc5044_mlops.dataset import slug

app = typer.Typer(help="Limpieza y enriquecimiento del dataset MiBici.")

def cargar_nomenclatura(ruta: Path) -> pd.DataFrame | None:
    """Catálogo de estaciones. Ojo: es una foto del estado ACTUAL del sistema,
    no un histórico, así que estaciones retiradas no aparecen."""
    if not ruta.exists():
        return None
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

    cols = {slug(c): c for c in df.columns}
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


def eliminar_duplicados_exactos(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva viajes con el mismo id si difieren en otra columna."""
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def limpiar(entrada: Path, nomenclatura: Path, salida: Path, reporte: Path,
            par: dict | None = None) -> dict:
    """Aplica la regla de limpieza y devuelve el reporte de métricas."""
    par = par if par is not None else cargar_params("limpiar")
    df = pd.read_parquet(entrada)
    n0 = len(df)
    rep: dict = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entrada": str(entrada),
        "params": par,
        "viajes_entrada": n0,
    }
    print(f"entrada: {n0:,} viajes ({df['periodo'].min()} a {df['periodo'].max()})")

    # -- 1. duplicados exactos: único caso en que sí se borra ---------------
    antes = len(df)
    df = eliminar_duplicados_exactos(df)
    rep["duplicados_eliminados"] = antes - len(df)
    print(f"duplicados exactos eliminados : {rep['duplicados_eliminados']:,}")

    # -- 2. bandera de validez por duración ---------------------------------
    dmin, dmax = par["duracion_min_valida"], par["duracion_max_valida"]
    duracion = pd.to_numeric(df["duracion_min"], errors="coerce")
    # Una duración nula NO es válida (defecto corregido en el PR #2)
    df["viaje_valido"] = duracion.notna() & duracion.between(dmin, dmax, inclusive="both")

    df["duracion_faltante"] = duracion.isna()
    df["duracion_corta"] = duracion < dmin
    df["duracion_larga"] = duracion > dmax
    rep["marcados_duracion_corta"] = int(df["duracion_corta"].sum())
    rep["marcados_duracion_larga"] = int(df["duracion_larga"].sum())
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
    nom = cargar_nomenclatura(Path(nomenclatura))
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
    df["dia_semana"] = ini.dt.day_of_week.astype("Int8")   # 0=lunes
    df["es_fin_semana"] = df["dia_semana"].isin([5, 6])
    df["mes"] = ini.dt.month.astype("Int8")
    df["fecha"] = ini.dt.date
    df["viaje_redondo"] = df["origen_id"] == df["destino_id"]
    rep["viajes_redondos"] = int(df["viaje_redondo"].sum())

    # -- 6. salida -----------------------------------------------------------
    sal = Path(salida)
    sal.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(sal, index=False, compression="zstd")

    rep["viajes_salida"] = len(df)
    rep["pct_validos"] = round(100 * rep["viajes_validos"] / len(df), 4)
    rep["salida"] = str(sal)
    rep["mb_salida"] = round(sal.stat().st_size / 1048576, 1)

    rpt = Path(reporte)
    rpt.parent.mkdir(parents=True, exist_ok=True)
    rpt.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(df):,} viajes -> {sal} ({rep['mb_salida']} MB)")
    print(f"válidos para modelar: {rep['viajes_validos']:,} ({rep['pct_validos']} %)")
    print(f"reporte: {rpt}")
    return rep


@app.command()
def main(
    entrada: Path = typer.Option(INTERIM_DATA_DIR / "mibici_viajes.parquet"),
    nomenclatura: Path = typer.Option(RAW_DATA_DIR / "mibici" / "nomenclatura"),
    salida: Path = typer.Option(PROCESSED_DATA_DIR / "mibici_viajes.parquet"),
    reporte: Path = typer.Option(REPORTS_DIR / "limpieza.json"),
) -> None:
    """Marca viajes inválidos, anula edades imposibles y une el catálogo de estaciones."""
    limpiar(entrada, nomenclatura, salida, reporte)
    logger.success("Limpieza completa.")


if __name__ == "__main__":
    app()
