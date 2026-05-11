# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./

# Keep the expensive dependency layer independent from app/config/data edits.
# Preinstall CPU Torch so sentence-transformers does not pull CUDA wheels.
RUN python -c "import tomllib; data=tomllib.load(open('pyproject.toml','rb')); print('\n'.join(data['project']['dependencies']))" > /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3" \
    && pip install -r /tmp/requirements.txt

COPY backend ./backend
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps -e .

RUN mkdir -p data logs

EXPOSE 8008

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8008"]
