"""Exporta el esquema OpenAPI del servicio a un archivo JSON (documentación versionada).

Uso:
    python -m tc5044_mlops.api.openapi                 # docs/openapi.json
    python -m tc5044_mlops.api.openapi otra/ruta.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from tc5044_mlops.config import PROJ_ROOT


def exportar_openapi(destino: str | Path | None = None) -> Path:
    from tc5044_mlops.api.main import app

    destino = Path(destino) if destino else PROJ_ROOT / "docs" / "openapi.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    return destino


if __name__ == "__main__":
    print(exportar_openapi(sys.argv[1] if len(sys.argv) > 1 else None))
