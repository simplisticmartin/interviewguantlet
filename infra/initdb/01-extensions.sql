-- Runs once, on first boot of the Postgres volume.
-- Alembic also guards these, but creating them here means a fresh `docker compose up`
-- is immediately usable by tools that connect before migrations run.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
