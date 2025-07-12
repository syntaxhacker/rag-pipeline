#!/bin/bash
set -e  # Exit on any error

echo "=== RAG Pipeline Startup ==="
echo "PORT environment variable: ${PORT:-'not set'}"
echo "Using port: ${PORT:-8000}"
echo "Google API Key present: ${GOOGLE_API_KEY:+'Yes':'No'}"
echo "Starting uvicorn server..."
echo "=== End Startup Info ==="

# Test if we can import the app first
echo "Testing app import..."
python -c "
try:
    from app.main import app
    print('✓ App imported successfully')
except Exception as e:
    print(f'✗ App import failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

# Start uvicorn with proper error handling
echo "Executing: uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level debug"
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level debug