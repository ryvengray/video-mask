#!/bin/bash
set -e

BACKUP_DIR="${1:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/audit-backup-$TIMESTAMP.tar.gz"

echo "Creating backup..."
mkdir -p "$BACKUP_DIR"

# Backup Docker volumes using a temporary container
docker run --rm \
    -v video-audit-data:/data \
    -v video-audit-videos:/videos \
    -v video-audit-screenshots:/screenshots \
    -v "$(pwd)/$BACKUP_DIR:/backup" \
    alpine tar czf "/backup/audit-backup-$TIMESTAMP.tar.gz" \
    -C /data . -C /videos . -C /screenshots .

echo "Backup created: $BACKUP_FILE"
