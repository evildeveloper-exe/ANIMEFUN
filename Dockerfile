FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY anime_data.json .
COPY model_trainer.py .
COPY local_server.py .
COPY index.html .

EXPOSE 8080

CMD ["sh", "-c", "python model_trainer.py && gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 300 local_server:app"]
