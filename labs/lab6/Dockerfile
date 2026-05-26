FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Pre-generated data — run scripts/generate_data.py locally first
COPY data/train.csv          ./data/train.csv
COPY data/public_test.csv    ./data/public_test.csv
COPY data/private_labels.csv ./data/private_labels.csv

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
