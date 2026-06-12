# Classifier service: Python 3.12 + uv.
FROM python:3.12-slim

# Copy uv from the official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Application code.
COPY app ./app

# Use venv and fixed HF cache path.
ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]