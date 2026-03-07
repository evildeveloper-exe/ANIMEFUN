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

# Train ML models at build time
RUN python model_trainer.py

# Startup script: re-trains if models are missing, then launches gunicorn
RUN printf '#!/bin/bash\nset -e\nif [ ! -f "/app/models/anime_list.pkl" ]; then\n  echo "[START] Models missing — training..."\n  python model_trainer.py\nfi\necho "[START] Launching gunicorn..."\nexec gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 2 --timeout 120 local_server:app\n' > /app/start.sh && chmod +x /app/start.sh

EXPOSE 8080

CMD ["/app/start.sh"]
