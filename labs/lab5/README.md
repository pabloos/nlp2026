# Lab 5 — Twitter Sentiment Server

Mini-Kaggle competitivo para clasificación de sentimiento en tweets con modelos pre-entrenados de Hugging Face.

## Quickstart (local)

```bash
cd labs/lab5
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

El primer arranque descarga `tweet_eval/sentiment` (~30 s) y genera los CSV en `data/`.

- Leaderboard: http://localhost:8000/
- API docs: http://localhost:8000/docs
- Datos para alumnos: http://localhost:8000/data/train.csv · /data/public_test.csv

## Dataset

`tweet_eval` sentiment (SemEval 2017) — solo tweets en inglés con etiqueta positiva o negativa (se descartan los neutrales).

| Split | Tweets |
|-------|--------|
| Train (visible) | 2 000 |
| Test (oculto) | 300 (150 pos + 150 neg) |

## Antes de la sesión

1. Arrancar el servidor y esperar a que genere los datos.
2. Reemplazar `SERVER_URL = "http://localhost:8000"` en el notebook con la URL real (Cloud Run, ngrok, etc.).
3. Compartir el notebook `lab5_twitter_sentiment_student.ipynb` con los alumnos.

## Deploy en Cloud Run

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/nlp-lab5
gcloud run deploy nlp-lab5 \
  --image gcr.io/PROJECT_ID/nlp-lab5 \
  --platform managed --region europe-southwest1 \
  --allow-unauthenticated --memory 512Mi
```

> Los datos se generan en el primer arranque si no están incluidos en la imagen.  
> Para incluirlos: genera los CSV localmente primero y el Dockerfile los copia.

## Techo de rendimiento

| Modelo | F1 (pos) estimado |
|--------|-------------------|
| DistilBERT SST-2 (baseline) | ~0.72 |
| CardiffNLP twitter-roberta-base-sentiment-latest | ~0.87 |
| BERTweet fine-tuned | ~0.85 |
