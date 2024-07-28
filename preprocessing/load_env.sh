#!/bin/sh
set -e

# Load environment variables from /data/.env
if [ -f /data/.env ]; then
    export $(cat /data/.env | xargs)
fi

# Execute the main command
exec "$@"