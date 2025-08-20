#!/bin/sh

set -e

echo "Checking if migration already ran..."

# Define the lock file path
LOCK_FILE="migration/.migration_locked"

echo "Checking if migration already ran..."

# Check if the lock file exists
if [ -f "$LOCK_FILE" ]; then
  echo "Migration lock file found. Exiting to prevent duplicate migration."
  exit 0
fi

echo "Run uv sync for installing new dependencies"
uv sync --frozen --no-dev --compile-bytecode --python=/usr/local/bin/python3.12

echo "Making data_collection/converter/id2index.json file"
python3 migration/quick_convert_id2index.py

echo "Making migration_data/id2index.json successfully"

echo "Starting app migration"
echo "---Starting embedding migration"
python migration/embedding_migration.py --file_path "migration_data/CLIP_convnext_xxlarge_laion2B_s34B_b82K_augreg_soup_clip_embeddings.pt"

echo "---Starting keyframe migration"
python migration/keyframe_migration.py --file_path "data_collection/converter/id2index.json"

echo "---Predownload open-clip model's weight"
python migration/download_open_clip_model.py

touch $LOCK_FILE

echo "End app migration"
