# Dockerfile for FastAPI inference service (data_science_final)
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ && rm -rf /var/lib/apt/lists/*

# Устанавливаем зависимости для ML/serving
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

# Копируем код и модели
COPY src/ ./src/
COPY models/ ./models/

# Переменные окружения
ENV MODEL_PATH=models/best_model_full.pkl
ENV META_PATH=models/best_model_full_meta.json
ENV THRESHOLD=0.1

EXPOSE 8000
CMD ["uvicorn","src.service:app","--host","0.0.0.0","--port","8000"]

