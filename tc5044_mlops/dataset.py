"""
Descarga y consolidación del dataset de datos abiertos de MiBici (Guadalajara).

Los CSV mensuales son archivos estáticos públicos enlazados desde
https://www.mibici.net/es/datos-abiertos/ (sin registro ni API key).

Uso (desde la raíz del proyecto):
    python -m tc5044_mlops.dataset listar
    python -m tc5044_mlops.dataset descargar            # rango de params.yaml
    python -m tc5044_mlops.dataset descargar --desde 2025-01 --hasta 2026-08
    python -m tc5044_mlops.dataset consolidar

Antes vivía en scripts/mibici_dataset.py; se movió al paquete en la Fase 2
para que la lógica sea importable, probada y parametrizada desde params.yaml.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import time
import unicodedata
import urllib.parse

from loguru import logger
import requests
import typer

from tc5044_mlops.config import INTERIM_DATA_DIR, RAW_DATA_DIR, cargar_params

app = typer.Typer(help="Descarga y consolidación del dataset MiBici.")

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


def descargar_dataset(out: Path, desde: str | None = None, hasta: str | None = None) -> Path:
    """Descarga los CSV de viajes (filtrados por periodo) y la nomenclatura.

    Devuelve la ruta del manifiesto con los MD5 de cada archivo.
    """
    out = Path(out)
    cat = listar_urls()
    manifiesto = []

    # Filtro opcional por rango de periodos (la nomenclatura nunca se filtra)
    if desde or hasta:
        desde = desde or "0000-00"
        hasta = hasta or "9999-99"
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
    return mf


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


def slug(s: str) -> str:
    """Normaliza un encabezado: minúsculas, sin acentos y con guiones bajos."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def leer_csv(ruta: Path):
    """Lee un CSV probando las codificaciones que ha usado MiBici."""
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
    cols = {slug(c): c for c in df.columns}
    mapeo = {}
    for canon, alias in SINONIMOS.items():
        for clave, original in cols.items():
            if clave in alias:
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


def consolidar_dataset(entrada: Path, salida: Path):
    """Une los CSV mensuales en un único dataset con esquema normalizado."""
    import pandas as pd

    entrada = Path(entrada)
    dir_viajes = entrada / "viajes" if (entrada / "viajes").is_dir() else entrada
    archivos = sorted(p for p in dir_viajes.glob("*.csv") if "datos_abiertos" in p.name.lower())
    if not archivos:
        sys.exit(f"No hay CSV de viajes en {dir_viajes}")

    partes = []
    for p in archivos:
        print(f"  leyendo {p.name}")
        partes.append(_normalizar(leer_csv(p), p))

    df = pd.concat(partes, ignore_index=True)
    df = df.sort_values("inicio_viaje", kind="mergesort").reset_index(drop=True)
    df["duracion_min"] = (df["fin_viaje"] - df["inicio_viaje"]).dt.total_seconds() / 60

    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    if salida.suffix == ".parquet":
        df.to_parquet(salida, index=False, compression="zstd")
    else:
        df.to_csv(salida, index=False)

    print(f"\n{len(df):,} viajes  |  {df['periodo'].min()} a {df['periodo'].max()}")
    print(f"{salida}  ({salida.stat().st_size/1048576:.0f} MB)")
    print("\nNulos por columna:")
    print(df.isna().sum().to_string())
    return df


# --------------------------------------------------------------------------
# 4. Interfaz de línea de comandos
# --------------------------------------------------------------------------
@app.command("listar")
def cli_listar() -> None:
    """Imprime las URLs de CSV encontradas en la página de datos abiertos."""
    for grupo, urls in listar_urls().items():
        print(f"\n== {grupo} ({len(urls)}) ==")
        print("\n".join(urls))


@app.command("descargar")
def cli_descargar(
    out: Path = typer.Option(RAW_DATA_DIR / "mibici", help="Carpeta de destino"),
    desde: str | None = typer.Option(None, help="Primer periodo YYYY-MM (por defecto params.yaml)"),
    hasta: str | None = typer.Option(None, help="Último periodo YYYY-MM (por defecto params.yaml)"),
) -> None:
    """Descarga los CSV del rango indicado (incremental: omite los que ya existen)."""
    rango = cargar_params("descargar")
    descargar_dataset(out, desde or rango["desde"], hasta or rango["hasta"])
    logger.success("Descarga completa.")


@app.command("consolidar")
def cli_consolidar(
    entrada: Path = typer.Option(RAW_DATA_DIR / "mibici", "--in", help="Carpeta con los CSV"),
    out: Path = typer.Option(INTERIM_DATA_DIR / "mibici_viajes.parquet", help="Parquet de salida"),
) -> None:
    """Une los CSV mensuales en un Parquet con esquema normalizado."""
    consolidar_dataset(entrada, out)
    logger.success("Consolidación completa.")


if __name__ == "__main__":
    app()
