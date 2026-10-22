#!/usr/bin/env python3
"""
Descarga y consolidación del dataset completo de datos abiertos de MiBici
(Guadalajara), sin registro ni API key: los CSV mensuales son archivos
estáticos públicos enlazados desde https://www.mibici.net/es/datos-abiertos/

Uso:
    python mibici_dataset.py listar
    python mibici_dataset.py descargar --out data/raw/mibici
    python mibici_dataset.py descargar --desde 2025-01 --hasta 2026-08
    python mibici_dataset.py consolidar --in data/raw/mibici --out data/interim/mibici_viajes.parquet

Requisitos: requests, pandas, pyarrow
    uv pip install requests pandas pyarrow
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.parse
from pathlib import Path

import requests

PAGINA = "https://www.mibici.net/es/datos-abiertos/"
UA = "Mozilla/5.0 (X11; Linux x86_64) mibici-dataset-builder/1.0 (uso académico)"
TIMEOUT = 120


# --------------------------------------------------------------------------
# 1. Descubrimiento de URLs
# --------------------------------------------------------------------------
def listar_urls() -> dict[str, list[str]]:
    """Scrapea la página de datos abiertos y devuelve los CSV por categoría.

    Los IDs de carpeta (/site/assets/files/<id>/) son arbitrarios y NO siguen
    un patrón derivable del año/mes, por eso hay que leerlos de la página en
    lugar de construirlos.
    """
    r = requests.get(PAGINA, headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    hrefs = set(re.findall(r'href=[\'"]([^\'"]+?\.csv)[\'"]', r.text, flags=re.I))
    urls = sorted(urllib.parse.urljoin(PAGINA, h) for h in hrefs)

    viajes = [u for u in urls if "datos_abiertos" in u.lower()]
    nomenclatura = [u for u in urls if "nomenclatura" in u.lower()]
    otros = [u for u in urls if u not in viajes and u not in nomenclatura]
    return {"viajes": viajes, "nomenclatura": nomenclatura, "otros": otros}


# --------------------------------------------------------------------------
# 2. Descarga incremental
# --------------------------------------------------------------------------
def descargar(url: str, destino: Path, reintentos: int = 3) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and destino.stat().st_size > 0:
        print(f"  ya existe, omito: {destino.name}")
        return destino

    tmp = destino.with_suffix(destino.suffix + ".part")
    for intento in range(1, reintentos + 1):
        try:
            with requests.get(
                url, headers={"User-Agent": UA}, timeout=TIMEOUT, stream=True
            ) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", 0))
                escrito = 0
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        escrito += len(chunk)
            if total and escrito != total:
                raise IOError(f"tamaño incompleto {escrito}/{total}")
            tmp.replace(destino)
            print(f"  OK {destino.name} ({escrito/1048576:.1f} MB)")
            return destino
        except Exception as exc:  # noqa: BLE001
            print(f"  intento {intento}/{reintentos} falló ({exc})", file=sys.stderr)
            time.sleep(2 * intento)
    raise RuntimeError(f"no se pudo descargar {url}")


def periodo_de(nombre: str) -> str | None:
    """Extrae 'YYYY-MM' del nombre de archivo, o None si no aplica."""
    m = re.search(r"(\d{4})[_-](\d{2})", nombre)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def cmd_descargar(args) -> None:
    out = Path(args.out)
    cat = listar_urls()
    manifiesto = []

    # Filtro opcional por rango de periodos (la nomenclatura nunca se filtra)
    if args.desde or args.hasta:
        desde = args.desde or "0000-00"
        hasta = args.hasta or "9999-99"
        antes = len(cat["viajes"])
        cat["viajes"] = [
            u for u in cat["viajes"]
            if (p := periodo_de(u.rsplit("/", 1)[-1])) and desde <= p <= hasta
        ]
        print(f"Filtro {desde}..{hasta}: {len(cat['viajes'])} de {antes} meses")

    for grupo, urls in cat.items():
        if not urls:
            continue
        print(f"\n== {grupo} ({len(urls)} archivos) ==")
        for url in urls:
            nombre = url.rsplit("/", 1)[-1]
            ruta = out / ("nomenclatura" if grupo == "nomenclatura" else "viajes") / nombre
            descargar(url, ruta)
            h = hashlib.md5(ruta.read_bytes()).hexdigest()
            manifiesto.append(
                {"url": url, "archivo": str(ruta), "bytes": ruta.stat().st_size, "md5": h}
            )

    mf = out / "manifiesto.json"
    mf.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    total_mb = sum(m["bytes"] for m in manifiesto) / 1048576
    print(f"\n{len(manifiesto)} archivos, {total_mb:.0f} MB. Manifiesto: {mf}")


# --------------------------------------------------------------------------
# 3. Consolidación con esquema normalizado
# --------------------------------------------------------------------------
# MiBici ha cambiado los encabezados y la codificación a lo largo de los años
# (español/inglés, con y sin acentos, columna índice fantasma). Normalizamos
# por nombre canónico y, si no hay coincidencia, por posición.
CANON = ["viaje_id", "usuario_id", "genero", "anio_nacimiento",
         "inicio_viaje", "fin_viaje", "origen_id", "destino_id"]

SINONIMOS = {
    "viaje_id": {"viaje_id", "trip_id", "aviaje_id", "id_viaje"},
    "usuario_id": {"usuario_id", "user_id", "id_usuario"},
    "genero": {"genero", "gender", "sexo"},
    "anio_nacimiento": {"ano_de_nacimiento", "anio_de_nacimiento", "birth_year",
                        "year_of_birth", "ano_nacimiento"},
    "inicio_viaje": {"inicio_del_viaje", "trip_start", "hora_de_retiro", "inicio_viaje"},
    "fin_viaje": {"fin_del_viaje", "trip_end", "hora_de_arribo", "fin_viaje"},
    "origen_id": {"origen_id", "origin_id", "id_origen"},
    "destino_id": {"destino_id", "destination_id", "id_destino"},
}


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def _leer_csv(ruta: Path):
    import pandas as pd

    ultimo = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(ruta, encoding=enc, low_memory=False)
        except UnicodeDecodeError as exc:  # noqa: PERF203
            ultimo = exc
    raise ultimo  # type: ignore[misc]


def _normalizar(df, ruta: Path):
    import pandas as pd

    df = df.loc[:, ~df.columns.astype(str).str.match(r"Unnamed")]
    cols = {_slug(c): c for c in df.columns}
    mapeo = {}
    for canon, alias in SINONIMOS.items():
        for slug, original in cols.items():
            if slug in alias:
                mapeo[original] = canon
                break

    faltan = [c for c in CANON if c not in mapeo.values()]
    if faltan:
        if len(df.columns) == len(CANON):  # fallback posicional
            print(f"  [aviso] {ruta.name}: encabezados no reconocidos "
                  f"{list(df.columns)} -> mapeo posicional")
            df.columns = CANON
            mapeo = {}
        else:
            raise ValueError(f"{ruta.name}: faltan columnas {faltan}; "
                             f"encabezados reales: {list(df.columns)}")
    if mapeo:
        df = df.rename(columns=mapeo)[CANON]

    for c in ("viaje_id", "usuario_id", "origen_id", "destino_id", "anio_nacimiento"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ("inicio_viaje", "fin_viaje"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["genero"] = df["genero"].astype("string").str.strip().str.upper()

    m = re.search(r"(\d{4})[_-](\d{2})", ruta.name)
    df["periodo"] = f"{m.group(1)}-{m.group(2)}" if m else pd.NA
    df["archivo_origen"] = ruta.name
    return df


def cmd_consolidar(args) -> None:
    import pandas as pd

    entrada = Path(getattr(args, "in"))
    dir_viajes = entrada / "viajes" if (entrada / "viajes").is_dir() else entrada
    archivos = sorted(p for p in dir_viajes.glob("*.csv") if "datos_abiertos" in p.name.lower())
    if not archivos:
        sys.exit(f"No hay CSV de viajes en {dir_viajes}")

    partes = []
    for p in archivos:
        print(f"  leyendo {p.name}")
        partes.append(_normalizar(_leer_csv(p), p))

    df = pd.concat(partes, ignore_index=True)
    df = df.sort_values("inicio_viaje", kind="mergesort").reset_index(drop=True)
    df["duracion_min"] = (df["fin_viaje"] - df["inicio_viaje"]).dt.total_seconds() / 60

    salida = Path(args.out)
    salida.parent.mkdir(parents=True, exist_ok=True)
    if salida.suffix == ".parquet":
        df.to_parquet(salida, index=False, compression="zstd")
    else:
        df.to_csv(salida, index=False)

    print(f"\n{len(df):,} viajes  |  {df['periodo'].min()} a {df['periodo'].max()}")
    print(f"{salida}  ({salida.stat().st_size/1048576:.0f} MB)")
    print("\nNulos por columna:")
    print(df.isna().sum().to_string())


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("listar", help="imprime las URLs encontradas")

    d = sub.add_parser("descargar", help="descarga los CSV (todos o un rango)")
    d.add_argument("--out", default="data/raw/mibici")
    d.add_argument("--desde", metavar="YYYY-MM", help="primer periodo a descargar")
    d.add_argument("--hasta", metavar="YYYY-MM", help="último periodo a descargar")
    d.set_defaults(func=cmd_descargar)

    c = sub.add_parser("consolidar", help="une los CSV en un solo dataset normalizado")
    c.add_argument("--in", default="data/raw/mibici")
    c.add_argument("--out", default="data/interim/mibici_viajes.parquet")
    c.set_defaults(func=cmd_consolidar)

    args = ap.parse_args()
    if args.cmd == "listar":
        cat = listar_urls()
        for grupo, urls in cat.items():
            print(f"\n== {grupo} ({len(urls)}) ==")
            print("\n".join(urls))
        return
    args.func(args)


if __name__ == "__main__":
    main()
