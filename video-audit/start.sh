#!/bin/bash
set -e

echo "Starting Video Audit..."

# Create .env if not exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo ".env file created from .env.example"
    echo "Please edit .env to set your CONTROLLER_URL"
fi

docker compose up --build -d

echo ""
echo "Video Audit is running at http://localhost:8080"
echo ""
echo "Data is persisted in Docker volumes:"
echo "  - video-audit-data      (SQLite database)"
echo "  - video-audit-videos     (downloaded videos)"
echo "  - video-audit-screenshots (review screenshots)"
echo ""
echo "To view logs: docker compose logs -f"
echo "To stop: docker compose down"
echo "To backup: docker run --rm -v video-audit-data:/data -v \$(pwd):/backup alpine tar czf /backup/audit-backup-\$(date +%Y%m%d).tar.gz -C /data ."
