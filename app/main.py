from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import sys
from .config import DATASET_CONFIGS
# Lazy imports to avoid blocking startup
# from .pipeline import RAGPipeline  # Will import when needed
# import umap  # Will import when needed for visualization
# import plotly.express as px  # Will import when needed for visualization
# import plotly.graph_objects as go  # Will import when needed for visualization
# from plotly.subplots import make_subplots  # Will import when needed for visualization
# import numpy as np  # Will import when needed for visualization
# from sklearn.preprocessing import normalize  # Will import when needed for visualization
# import pandas as pd  # Will import when needed for visualization
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Pipeline API", description="Multi-dataset RAG API", version="1.0.0")

# Initialize pipelines for all datasets
pipelines = {}
google_api_key = os.getenv("GOOGLE_API_KEY")

logger.info(f"Starting RAG Pipeline API")
logger.info(f"Port from env: {os.getenv('PORT', 'Not set - will use 8000')}")
logger.info(f"Google API Key present: {'Yes' if google_api_key else 'No'}")
logger.info(f"Available datasets: {list(DATASET_CONFIGS.keys())}")

# Don't load datasets during startup - do it asynchronously after server starts
logger.info("RAG Pipeline API is ready to serve requests - datasets will load in background")

# Visualization function disabled to speed up startup
# def create_3d_visualization(pipeline):
#     ... (commented out for faster startup)

class Question(BaseModel):
    text: str
    dataset: str = "settings-dataset"  # Default dataset

@app.post("/answer")
async def get_answer(question: Question):
    try:
        # Check if any pipelines are loaded
        if not pipelines:
            return {
                "answer": "RAG Pipeline is running but datasets are still loading in the background. Please try again in a moment, or check /health for loading status.",
                "dataset": question.dataset,
                "status": "datasets_loading"
            }
        
        # Select the appropriate pipeline based on dataset
        if question.dataset not in pipelines:
            raise HTTPException(status_code=400, detail=f"Dataset '{question.dataset}' not available. Available datasets: {list(pipelines.keys())}")
        
        selected_pipeline = pipelines[question.dataset]
        answer = selected_pipeline.answer_question(question.text)
        return {"answer": answer, "dataset": question.dataset}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datasets")
async def list_datasets():
    """List all available datasets"""
    return {"datasets": list(pipelines.keys())}

async def load_datasets_background():
    """Load datasets in background after server starts"""
    global pipelines
    if google_api_key:
        # Import RAGPipeline only when needed
        from .pipeline import RAGPipeline
        for dataset_name in DATASET_CONFIGS.keys():
            try:
                logger.info(f"Loading dataset: {dataset_name}")
                pipeline = RAGPipeline.from_preset(
                    google_api_key=google_api_key,
                    preset_name=dataset_name
                )
                pipelines[dataset_name] = pipeline
                logger.info(f"Successfully loaded {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {e}")
        logger.info(f"Background loading complete - {len(pipelines)} datasets loaded")
    else:
        logger.warning("No Google API key provided - running in demo mode without datasets")

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application startup complete")
    logger.info(f"Server should be running on port: {os.getenv('PORT', '8000')}")
    
    # Start loading datasets in background (non-blocking)
    import asyncio
    asyncio.create_task(load_datasets_background())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutting down")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"status": "ok", "message": "RAG Pipeline API", "version": "1.0.0", "datasets": list(pipelines.keys())}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check called")
    loading_status = "complete" if len(pipelines) == len(DATASET_CONFIGS) else "loading"
    return {
        "status": "healthy", 
        "datasets_loaded": len(pipelines), 
        "total_datasets": len(DATASET_CONFIGS),
        "loading_status": loading_status,
        "port": os.getenv('PORT', '8000')
    }

