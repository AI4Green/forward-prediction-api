FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libxrender1 \
    libxext6 \
    libsm6 \
    libx11-6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install uv && uv sync --frozen

COPY app/ ./app/
COPY trained_models/ ./trained_models/

ENV USE_CUDA_IF_AVAILABLE=false
ENV PYTHON_ENV="production"

ENV AUTH_ENABLED=false
ENV API_KEY="API-Key"
ENV API_KEY_HEADER="X-API-Key"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]