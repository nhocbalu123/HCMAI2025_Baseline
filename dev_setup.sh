#! bin/sh

mkdir -p migration_data

mkdir -p data_collection/{converter keyframe}

chmod +x migration/app_migration.sh

chmod +x entrypoint.sh

docker build -t hieudev89/rag_app .

docker compose up -d
