#!/bin/bash
set -e  # Exit on any error

echo "=== RAG Pipeline Startup ==="
echo "PORT environment variable: ${PORT:-'not set'}"
echo "Using port: ${PORT:-8000}"
echo "Google API Key present: ${GOOGLE_API_KEY:+'Yes':-'No'}"
echo "Starting uvicorn server..."
echo "=== End Startup Info ==="

# Start uvicorn with proper error handling
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info --timeout-keep-alive 0