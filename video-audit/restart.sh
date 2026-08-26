#!/bin/bash
set -e

echo "Restarting Video Audit..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Copying from .env.example"
    cp .env.example .env
fi

echo "Rebuilding and restarting..."
docker compose down
docker compose build --no-cache
docker compose up -d

echo ""
echo "Video Audit restarted at http://localhost:8080"
echo ""
echo "View logs: docker compose logs -f"
