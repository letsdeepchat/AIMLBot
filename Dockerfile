# ---- UI build ---------------------------------------------------------------
FROM node:22-alpine AS ui
WORKDIR /ui
COPY frontend/package*.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- API, worker and MLflow server share this image -------------------------
FROM python:3.12-slim
# libgomp1: OpenMP runtime LightGBM needs
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./
COPY --from=ui /ui/dist ./static
ENV QUEST_STATIC_DIR=/app/static \
    QUEST_DATA_DIR=/data \
    PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
