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

# Debug the uvicorn command
echo "Current directory: $(pwd)"
echo "Python path: $PYTHONPATH"
echo "Contents of current directory:"
ls -la
echo "Contents of app directory:"
ls -la app/
echo "Testing Python import:"
python -c "import app.main; print('Import successful')" || echo "Import failed"
echo "Starting uvicorn..."
uvicorn app.main_simple:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info