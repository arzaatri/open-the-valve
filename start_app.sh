#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

docker compose up -d
docker compose exec -T postgres sh -c 'until pg_isready -U otv; do sleep 1; done'

uv run alembic -c src/open_the_valve/db/migrations/alembic.ini upgrade head

echo "open-the-valve is up: postgres on localhost:5433 (db=open_the_valve, user=otv)"
