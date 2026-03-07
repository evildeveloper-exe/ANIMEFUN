FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY anime_data.json .
COPY model_trainer.py .
COPY local_server.py .
COPY AnimeSensei.html .

# Train ML models at build time so startup is instant
RUN python model_trainer.py

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "local_server:app"]
