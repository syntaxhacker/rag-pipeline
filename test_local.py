#!/usr/bin/env python3
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Test environment variables
print("=== Environment Variables ===")
print(f"GOOGLE_API_KEY: {'✓' if os.getenv('GOOGLE_API_KEY') else '✗'}")
print(f"HF_TOKEN: {'✓' if os.getenv('HF_TOKEN') else '✗'}")
print(f"PORT: {os.getenv('PORT', '8000')}")

# Test import
try:
    from app.main import app, pipelines
    print(f"\n=== App Status ===")
    print(f"App loaded: ✓")
    print(f"Pipelines loaded: {len(pipelines)}")
    print(f"Available datasets: {list(pipelines.keys())}")
    
    # Test if we can run the server
    print(f"\n=== Ready to start server ===")
    print("Run: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    
except Exception as e:
    print(f"\n=== Error ===")
    print(f"Failed to load app: {e}")
    import traceback
    traceback.print_exc()