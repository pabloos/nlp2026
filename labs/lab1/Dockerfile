FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY app/ ./app/

# Dataset files — private_labels.csv is included here (only in the server image)
COPY data/train.csv        ./data/train.csv
COPY data/public_test.csv  ./data/public_test.csv
COPY data/private_labels.csv ./data/private_labels.csv

EXPOSE 8080

# Cloud Run injects PORT; fall back to 8080 for local use
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
