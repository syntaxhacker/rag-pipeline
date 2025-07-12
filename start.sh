#!/bin/bash
echo "=== RAG Pipeline Startup ==="
echo "PORT environment variable: ${PORT:-'not set'}"
echo "Using port: ${PORT:-8000}"
echo "Starting uvicorn server..."
echo "=== End Startup Info ==="

exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info