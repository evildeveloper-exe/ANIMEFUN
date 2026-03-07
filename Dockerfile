FROM docker.io/library/python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY anime_data.json .
COPY model_trainer.py .
COPY local_server.py .
COPY AnimeSensei.html .

EXPOSE 8080

# --timeout 300 gives the auto-trainer 5 minutes to complete on first boot
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "local_server:app"]
