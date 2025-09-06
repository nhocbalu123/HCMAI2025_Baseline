#!/bin/sh

set -e
echo 'Syncing dependencies...'
uv sync --frozen --no-dev --compile-bytecode --python=/usr/local/bin/python3.12
# chown -R webappnonroot:webappnonroot /app/data_collection
# chown -R 777 /app/data_collection
echo 'Starting development server...'
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
