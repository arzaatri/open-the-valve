#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -f .streamlit.pid ]; then
    kill "$(cat .streamlit.pid)" 2>/dev/null || true
    rm .streamlit.pid
fi

docker compose down
