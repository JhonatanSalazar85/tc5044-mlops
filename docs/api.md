# Servicio de predicción (FastAPI)

Servicio REST que expone el modelo campeón de demanda de MiBici. Código en
`tc5044_mlops/api/` (`esquemas.py`: Pydantic, `servicio.py`: carga del modelo y preparación de
variables, `main.py`: endpoints). El esquema OpenAPI completo está versionado en
[`docs/openapi.json`](openapi.json) y hay una colección de Postman en
[`docs/postman_mibici_api.json`](postman_mibici_api.json).

## Modelo que se sirve

| Concepto | Valor |
|---|---|
| Modelo registrado | `mibici-demanda-estacion-hora` (MLflow Model Registry) |
| Alias | `@champion` → versión 1 (`hgb_optuna`) |
| Ruta del artefacto | `models:/mibici-demanda-estacion-hora/1` |
| Copia local (la que usa el servicio) | `models/champion/` (generada con `make exportar`, versionada con DVC) |
| Ficha | `models/champion/model_info.json` (run_id, métricas, SHA-256, entorno) |

El servicio **no** se conecta al servidor de MLflow: carga la copia exportada. La variable de
entorno `MODELO_DIR` cambia la carpeta (por ejemplo, `models/challenger`).

## Ejecutar

```bash
make api                     # uvicorn con recarga en http://127.0.0.1:8000
# o
uvicorn tc5044_mlops.api.main:app --host 0.0.0.0 --port 8000
```

Documentación interactiva: <http://127.0.0.1:8000/docs> (Swagger UI) y
<http://127.0.0.1:8000/redoc>. El esquema se regenera con `make openapi`.

## Endpoints

| Método | Ruta | Respuesta | Descripción |
|---|---|---|---|
| GET | `/health` | 200 `Salud` / 503 | Estado del servicio; 503 si el modelo no cargó. Lo usa el `HEALTHCHECK` del contenedor. |
| GET | `/model-info` | 200 `InfoModelo` | URI del Registry, versión, run_id, métricas, hiperparámetros, SHA-256 del modelo. |
| POST | `/predict` | 200 `RespuestaPrediccion` / 422 / 503 | Predicción por lote (1 a 5 000 registros). |

Cada respuesta trae el encabezado `X-Tiempo-Proceso-ms`.

## Esquema de entrada (`POST /predict`)

```json
{
  "registros": [
    {
      "origen_id": 51,
      "fecha_hora": "2026-08-31T08:00:00",
      "demanda_lag_1h": 18,
      "demanda_lag_2h": 9,
      "demanda_lag_24h": 31,
      "demanda_lag_168h": 28,
      "demanda_media_3h": 10.33,
      "demanda_media_24h": 11.21,
      "demanda_media_168h": 10.64
    }
  ]
}
```

| Campo | Tipo | Obligatorio | Regla | Significado |
|---|---|---|---|---|
| `origen_id` | entero | sí | > 0 | Estación de retiro (catálogo MiBici). |
| `fecha_hora` | fecha-hora ISO 8601 | sí | se trunca a la hora; con zona horaria se convierte a `America/Mexico_City` | Hora a predecir. |
| `demanda_lag_1h`, `_2h`, `_24h`, `_168h` | número o `null` | no | 0 a 1 000 | Retiros hace 1, 2, 24 y 168 horas. |
| `demanda_media_3h`, `_24h`, `_168h` | número o `null` | no | 0 a 1 000 | Media de retiros de las 3, 24 y 168 horas previas. |

Las variables de calendario (`hora_sin`, `hora_cos`, `dia_sin`, `dia_cos`, `mes_sin`, `mes_cos`,
`es_fin_semana`, `hora`) **no se envían**: el servicio las calcula con la misma función del
entrenamiento (`features.agregar_calendario`) y con los mismos tipos de dato, de modo que la
API reproduce bit a bit las predicciones del pipeline por lotes (lo comprueba
`tests/test_api.py::test_predict_es_identico_al_pipeline_por_lotes`).

Campos desconocidos se rechazan (`extra="forbid"`): un nombre mal escrito no se convierte en
silencio en un rezago faltante.

## Esquema de salida

```json
{
  "modelo": "mibici-demanda-estacion-hora",
  "version_modelo": "1",
  "model_uri": "models:/mibici-demanda-estacion-hora/1",
  "version_api": "1.0.0",
  "n_registros": 1,
  "predicciones": [
    {"origen_id": 51, "fecha_hora": "2026-08-31T08:00:00", "prediccion": 17.86}
  ],
  "advertencias": []
}
```

`prediccion` está en bicicletas por hora y nunca es negativa (el valor del ejemplo es ilustrativo). `advertencias` informa, sin
bloquear la respuesta, si una estación no estaba en el entrenamiento o si hubo rezagos
faltantes (se imputan con la mediana del entrenamiento).

## Errores

| Código | Cuándo | Cuerpo |
|---|---|---|
| 422 | Entrada inválida (tipo, rango, campo faltante o desconocido, 0 o más de 5 000 registros) | `{"detail": [{"loc": ["body","registros",0,"origen_id"], "msg": "...", "type": "..."}]}` |
| 503 | El modelo no está cargado | `{"detail": "Modelo no disponible: ..."}` |
| 500 | Error inesperado al predecir | `{"detail": "Error al predecir: ..."}` |

## Ejemplos

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/model-info
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"registros":[{"origen_id":51,"fecha_hora":"2026-08-31T08:00:00","demanda_lag_1h":18,
       "demanda_lag_2h":9,"demanda_lag_24h":31,"demanda_lag_168h":28,"demanda_media_3h":10.33,
       "demanda_media_24h":11.21,"demanda_media_168h":10.64}]}'
```

```python
import httpx

r = httpx.post("http://127.0.0.1:8000/predict", json={"registros": [
    {"origen_id": 51, "fecha_hora": "2026-08-31T08:00:00", "demanda_lag_1h": 18}]})
print(r.json()["predicciones"], r.json()["advertencias"])
```
