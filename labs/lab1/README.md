# Lab 1 — Mini-Kaggle: Spam Detection

Servidor de scoring competitivo para el laboratorio de NLP clásico.  
Los alumnos entrenan TF-IDF + Logistic Regression en su notebook y envían predicciones aquí.

---

## Estructura

```
app/
  main.py                   # FastAPI app + lifespan
  dependencies.py           # Inyección de dependencias
  models/schemas.py         # Pydantic models
  repositories/
    base.py                 # Interfaz abstracta (fácil migración a Firestore)
    memory.py               # Implementación en memoria
  services/
    scoring.py              # Carga labels privados, calcula métricas
    leaderboard.py          # Rate limiting + registro de scores
  routes/
    submit.py               # POST /submit
    leaderboard.py          # GET /leaderboard  GET /  GET /history/{team}
data/
  train.csv                 # Disponible para alumnos
  public_test.csv           # Disponible para alumnos (sin labels)
  private_labels.csv        # PRIVADO — solo en el servidor
scripts/
  generate_data.py          # Genera los tres CSV
```

---

## Setup local

```bash
cd labs/lab1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
# El servidor descarga y prepara los datos automáticamente al arrancar.
```

Leaderboard en: http://localhost:8080  
Docs interactivos: http://localhost:8080/docs

---

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/submit` | Envía predicciones y recibe métricas |
| `GET` | `/leaderboard` | Leaderboard JSON |
| `GET` | `/` | Leaderboard HTML (auto-refresh 10 s) |
| `GET` | `/history/{team}` | Historial de submissions de un equipo |

### POST /submit — ejemplo

```bash
curl -X POST http://localhost:8080/submit \
  -H "Content-Type: application/json" \
  -d '{
    "team": "alpha",
    "predictions": [
      {"id": 0, "label": "ham"},
      {"id": 1, "label": "spam"}
    ]
  }'
```

Respuesta:
```json
{"f1": 0.923, "precision": 0.91, "recall": 0.936, "accuracy": 0.945, "rank": 1}
```

**Rate limit:** máximo 1 submit cada 30 segundos por equipo.

---

## Docker

```bash
# Build
docker build -t nlp-lab1 .

# Run local
docker run -p 8080:8080 nlp-lab1
```

---

## Deploy en Google Cloud Run

```bash
PROJECT_ID=tu-proyecto-gcp
IMAGE=gcr.io/$PROJECT_ID/nlp-lab1

# 1. Build y push
gcloud builds submit --tag $IMAGE

# 2. Deploy
gcloud run deploy nlp-lab1 \
  --image $IMAGE \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 512Mi

# 3. URL del servicio
gcloud run services describe nlp-lab1 --format "value(status.url)"
```

> **Nota:** Cloud Run es stateless — los scores se pierden al reiniciar la instancia.  
> Para persistencia real, migra `InMemorySubmissionRepository` → `FirestoreSubmissionRepository`  
> (misma interfaz `SubmissionRepository`, distinto backend).

---

## Migración a Firestore

1. Crea `app/repositories/firestore.py` implementando `SubmissionRepository`
2. En `app/main.py`, reemplaza:
   ```python
   # Antes:
   app.state.leaderboard_service = LeaderboardService(InMemorySubmissionRepository())
   # Después:
   app.state.leaderboard_service = LeaderboardService(FirestoreSubmissionRepository(collection="submissions"))
   ```
3. Añade `google-cloud-firestore` a `requirements.txt`

---

## Anti-cheating

- Rate limit: 1 submission / 30 s por equipo (HTTP 429 si se excede)
- `private_labels.csv` nunca expuesto por la API
- Validación de IDs: todos los samples de `public_test.csv` deben estar en el payload
