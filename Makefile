.PHONY: all setup permissions build up clean

# Define variables for common directories
MIGRATION_DATA_DIR = migration_data
DATA_COLLECTION_BASE_DIR = data_collection
CONVERTER_DIR = $(DATA_COLLECTION_BASE_DIR)/converter
KEYFRAME_DIR = $(DATA_COLLECTION_BASE_DIR)/keyframe

# Define variables for scripts
MIGRATION_SCRIPT = migration/app_migration.sh
ENTRYPOINT_SCRIPT = app_entrypoint.sh

# Define Docker image and compose file
DOCKER_IMAGE = hieudev89/rag_app
DOCKER_COMPOSE_FILE = docker-compose.yml

# Define HF cache dir
HF_HOME = hf_cache
HF_HOME_HUB = $(HF_HOME)/hub

# Checkpoints and migration_data
CHECKPOINTS = checkpoints
MIGRATION_DATA = migration_data
# Default target
all: folders_setup permissions build up

pre_setup: folders_setup permissions

## Setup Directories
# -----------------------------------------------------------------------------
folders_setup:
	@echo "Setting up necessary directories..."
	mkdir -p $(MIGRATION_DATA_DIR)
	mkdir -p $(CONVERTER_DIR)
	mkdir -p $(KEYFRAME_DIR)
	mkdir -p $(HF_HOME)
	mkdir -p $(CHECKPOINTS)
	mkdir -p $(MIGRATION_DATA)
	@echo "Directories created/ensured."

## Set Permissions
# -----------------------------------------------------------------------------
permissions:
	@echo "Setting executable permissions on scripts..."
	chmod +x $(MIGRATION_SCRIPT)
	chmod +x $(ENTRYPOINT_SCRIPT)
	@echo "Permissions set."

## Docker Build
# -----------------------------------------------------------------------------
build:
	@echo "Building Docker image $(DOCKER_IMAGE)..."
	docker build -t $(DOCKER_IMAGE) .
	@echo "Docker image built successfully."

## Docker Compose Up
# -----------------------------------------------------------------------------
up:
	@echo "Bringing up Docker services with Docker Compose..."
	docker compose -f $(DOCKER_COMPOSE_FILE) up -d
	@echo "Docker services are running in detached mode."

down:
	@echo "Bringing down Docker services with Docker Compose..."
	docker compose -f $(DOCKER_COMPOSE_FILE) down
	@echo "Docker services containers are removed"

## Clean Up Docker Resources
# -----------------------------------------------------------------------------
quick_clean:
	@echo "Quick clean up"
	docker compose -f $(DOCKER_COMPOSE_FILE) down -v
	rm -f migration/.migration_locked

clean:
	@echo "Stopping and removing Docker services and volumes..."
	docker compose -f $(DOCKER_COMPOSE_FILE) down --volumes --remove-orphans
	@echo "Remove .migration_locked"
	rm -f migration/.migration_locked
	@echo "Pruning unused Docker system resources..."
	docker system prune -a --volumes -f
	@echo "Docker resources cleaned."
