# Data Science Final Project — СберАвтоподписка

> Финальная работа по Data Science: подготовка данных, обучение моделей и развёртывание инференса как API на FastAPI в Docker.


## 📌 Содержание
- [Цель](#-цель)
- [Важное предупреждение](#-важное-предупреждение)
- [Структура проекта](#-структура-проекта)
- [Требования и установка](#-требования-и-установка)
- [Данные](#-данные)
- [Артефакты модели](#-артефакты-модели)
- [Локальный запуск (без Docker)](#-локальный-запуск-без-docker)
- [Сборка и запуск в Docker](#-сборка-и-запуск-в-docker)
- [API: эндпоинты и примеры](#-api-эндпоинты-и-примеры)
- [Проверка здоровья и мониторинг](#-проверка-здоровья-и-мониторинг)
- [Troubleshooting (WSL2/Docker Desktop)](#-troubleshooting-wsl2docker-desktop)
- [Среда разработки](#-среда-разработки)

---

## 🎯 Цель
Построить и развернуть модель, предсказывающую вероятность совершения пользователем целевого действия на сайте «СберАвтоподписка».

## ‼ Важное предупреждение

Для корректной работы проекта **обязательно наличие исходных файлов данных**:

- `data/raw/ga_hits.pkl`
- `data/raw/ga_sessions.pkl`

Эти файлы слишком большие и не хранятся в репозитории.  
Перед запуском проекта убедитесь, что они находятся в папке `data/raw/`.

---

## 📁 Структура проекта
```
data_science_final/
├─ data/
│  ├─ raw/
│  │  ├─ ga_hits.pkl
│  │  └─ ga_sessions.pkl
│  └─ processed/
├─ models/
│  ├─ best_model_full.pkl
│  └─ best_model_full_meta.json
├─ notebooks/
│  └─ eda.ipynb
├─ src/
│  ├─ predict.py
│  ├─ make_samples.py
│  └─ service.py        # FastAPI-приложение
├─ requirements.txt
├─ requirements-ml.txt  # зависимости для инференса
└─ Dockerfile
```

---

## 🔧 Требования и установка
**Python 3.12+**
```bash
pip install -r requirements.txt
pip install -r requirements-ml.txt
```

---

## 📊 Данные
Файлы:
- `data/raw/ga_hits.pkl`
- `data/raw/ga_sessions.pkl`

---

## 🧠 Артефакты модели
- `models/best_model_full.pkl` — сериализованная модель
- `models/best_model_full_meta.json` — метаинформация (порог, признаки и др.)

---

## ▶️ Локальный запуск (без Docker)
```bash
uvicorn src.service:app --host 0.0.0.0 --port 8000 --reload
```
Swagger UI: http://127.0.0.1:8000/docs  
Health: http://127.0.0.1:8000/healthz

---

## 🐳 Сборка и запуск в Docker
```bash
docker build -t ds-final-api:latest .
docker run --rm -p 8080:8000 ds-final-api:latest
```
Swagger UI: http://127.0.0.1:8080/docs  
Health: http://127.0.0.1:8080/healthz

---

## 🛰 API: эндпоинты и примеры
### POST /predict
```json
{
  "threshold": 0.1,
  "instances": [
    {
      "visit_number": 2,
      "utm_source": "google",
      "utm_medium": "cpc",
      "utm_campaign": "brand",
      "device_category": "mobile"
    }
  ]
}
```

### GET /healthz
Проверка сервиса.

---

## 🛠 Troubleshooting (WSL2/Docker Desktop)
1. `wsl --shutdown`
2. `net stop LxssManager && net start LxssManager`
3. Проверь `Get-Service com.docker.service`
4. Перезапусти Docker Desktop

---

## 💻 Среда разработки
- Python 3.12
- Docker Desktop 4.48.0
- WSL2 Ubuntu
- PyCharm / VS Code
