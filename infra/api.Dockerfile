FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY gauntlet ./gauntlet
COPY apps/api ./apps/api
COPY alembic ./alembic
COPY alembic.ini ./
COPY evals ./evals

RUN pip install --upgrade pip && pip install -e .

# Non-root: candidate-supplied text never gets root here.
RUN useradd --create-home --uid 10001 gauntlet && chown -R gauntlet /srv
USER gauntlet

EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn apps.api.main:app --host 0.0.0.0 --port 8000"]
