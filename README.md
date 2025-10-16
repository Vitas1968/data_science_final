# Data Science Final Project

Финальная работа по Data Science (СберАвтоподписка).

## Важное предупреждение

Для корректной работы проекта **обязательно наличие исходных файлов данных**:

- `data/raw/ga_hits.pkl`
- `data/raw/ga_sessions.pkl`

Эти файлы слишком большие и не хранятся в репозитории.  
Перед запуском проекта убедитесь, что они находятся в папке `data/raw/`.

## Установка зависимостей
- Базовые: `pip install -r requirements.txt`
- Расширенные (бустинг, балансировка): `pip install -r requirements-ml.txt`

## Структура проекта
- `data/` — датасеты (`ga_sessions.pkl`, `ga_hits.pkl`)
- `notebooks/` — анализ данных (EDA)
- `src/` — Python-скрипты (train.py, predict.py)
- `models/` — сохранённые модели
- `requirements.txt` — зависимости проекта

## Цель
Построить модель, предсказывающую вероятность совершения пользователем целевого действия на сайте «СберАвтоподписка».

🚀 Развёртывание модели в виде API-сервиса
Для инференса обученной модели создан сервис на FastAPI, который упакован в Docker-контейнер.

📦 Сборка Docker-образа
Из корня проекта выполните:

docker build -t ds-final-api:0.1.0 .
▶️ Запуск контейнера
docker run --rm -p 8080:8000 ds-final-api:0.1.0
После запуска сервис доступен по адресам:
👉 http://127.0.0.1:8080/docs — Swagger UI
👉 http://127.0.0.1:8080/healthz — проверка состояния

🧠 Пример запроса к API
POST /predict

{
  "threshold": 0.1,
  "instances": [
    {
      "visit_number": 2,
      "utm_source": "google",
      "utm_medium": "cpc",
      "utm_campaign": "brand",
      "utm_adcontent": "a1",
      "utm_keyword": "auto",
      "device_category": "mobile",
      "device_os": "android",
      "device_brand": "samsung",
      "device_model": "s23",
      "device_screen_resolution": "1080x1920",
      "device_browser": "chrome",
      "geo_country": "russia",
      "geo_city": "moscow",
      "visit_date": "2025-06-15"
    }
  ]
}
Пример ответа:

{
  "proba": [0.002],
  "pred": [0],
  "used_threshold": 0.1
}
⚙️ Содержимое контейнера
src/service.py — FastAPI-приложение
models/best_model_full.pkl — сериализованная модель
models/best_model_full_meta.json — метаинформация (порог, признаки и т.д.)
requirements-ml.txt — зависимости для инференса

🧩 Автор и среда
Разработано в рамках финального проекта курса Data Science.
Среда: Python 3.12, FastAPI, Docker 4.48.0, Windows 10 + WSL2.