# Lab 6 — Amazon Aspect Sentiment Server

Mini-Kaggle competitivo: aspect-based sentiment (calidad + entrega) en reseñas Amazon Electronics con un LLM local.

## Quickstart (local)

```bash
cd labs/lab6
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

El primer arranque descarga reviews Amazon vía streaming (~1-2 min) y genera los CSV.

- Leaderboard: http://localhost:8000/
- API docs: http://localhost:8000/docs

## Dataset

Amazon Reviews 2023 — Electronics (McAuley-Lab). Filtrado a reseñas que mencionan entrega/envío.

| Etiqueta | Fuente |
|----------|--------|
| `quality` | Rating ≥ 4 → pos, ≤ 2 → neg |
| `delivery` | Keyword rules sobre el texto de la reseña |

| Split | Reseñas |
|-------|---------|
| Train (visible) | 1 500 |
| Test (oculto) | 100 (25 por combinación quality×delivery) |

## Modelo para alumnos

Instala [Ollama](https://ollama.com) en los ordenadores de los alumnos y descarga el modelo:

```bash
ollama pull llama3.2:3b   # ~2 GB, funciona en CPU
# alternativa más ligera:
ollama pull llama3.2:1b   # ~0.7 GB, más rápido pero menos preciso
```

El notebook usa `http://localhost:11434` por defecto. Cambia `MODEL` en la celda de configuración al modelo que hayas distribuido.

## Métrica

**F1 global** = (F1 calidad + F1 entrega) / 2

El leaderboard muestra F1 global, F1 calidad y F1 entrega por separado para que los alumnos identifiquen dónde flaquea su prompt.

## Antes de la sesión

1. Correr `python scripts/generate_data.py` para generar los CSV localmente.
2. Revisar la distribución de etiquetas en `data/private_labels.csv`.
3. Iniciar el servidor y compartir la URL con los alumnos.
4. Distribuir el modelo Ollama (USB / red local) y el notebook `lab6_amazon_qa_student.ipynb`.

## Techo de rendimiento

| Enfoque | F1 global estimado |
|---------|-------------------|
| Prompt directo sin ejemplos | ~0.60-0.65 |
| Prompt con instrucciones detalladas | ~0.72-0.78 |
| Few-shot (2 ejemplos) | ~0.78-0.83 |
