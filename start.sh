#!/bin/bash
set -e  # Exit on any error

echo "=== RAG Pipeline Startup ==="
echo "PORT environment variable: ${PORT:-'not set'}"
echo "Using port: ${PORT:-8000}"
if [ -n "$GOOGLE_API_KEY" ]; then
    echo "Google API Key present: Yes"
else
    echo "Google API Key present: No"
fi
echo "Starting uvicorn server..."
echo "=== End Startup Info ==="

# Start uvicorn directly without extra debugging
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info