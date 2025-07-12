from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import sys
from .pipeline import RAGPipeline
from .config import DATASET_CONFIGS
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd
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

def create_3d_visualization(pipeline: RAGPipeline):
    # Create visualizations directory if it doesn't exist
    os.makedirs("/app/visualizations", exist_ok=True)
    
    # Get all documents using the correct API method
    documents = pipeline.document_store.filter_documents()
    
    # Extract embeddings and metadata
    embeddings = []
    metadata = []
    
    for doc in documents:
        if hasattr(doc, 'embedding') and doc.embedding is not None:
            embeddings.append(doc.embedding)
            try:
                content_dict = json.loads(doc.content)
                metadata.append({
                    'section': content_dict.get('section', 'unknown'),
                    'setting': content_dict.get('setting', 'unknown'),
                    'component_type': content_dict.get('component_type', 'unknown'),
                    'description': content_dict.get('description', '')[:100] + '...',
                    'tab': content_dict.get('tab', 'unknown')
                })
            except:
                metadata.append({
                    'section': 'unknown',
                    'setting': 'unknown',
                    'component_type': 'unknown',
                    'description': doc.content[:100] + '...',
                    'tab': 'unknown'
                })
    
    if not embeddings:
        print("No embeddings found in documents")
        return
        
    embeddings = np.array(embeddings)
    
    # Normalize embeddings
    embeddings_normalized = normalize(embeddings)
    
    # Create both 2D and 3D UMAP embeddings
    reducer_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_3d = reducer_3d.fit_transform(embeddings_normalized)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': embeddings_3d[:, 0],
        'y': embeddings_3d[:, 1],
        'z': embeddings_3d[:, 2],
        'section': [m['section'] for m in metadata],
        'setting': [m['setting'] for m in metadata],
        'component_type': [m['component_type'] for m in metadata],
        'description': [m['description'] for m in metadata],
        'tab': [m['tab'] for m in metadata]
    })
    
    # Create 3D visualization
    fig = go.Figure()
    
    # Add traces for each section
    for section in df['section'].unique():
        mask = df['section'] == section
        fig.add_trace(go.Scatter3d(
            x=df[mask]['x'],
            y=df[mask]['y'],
            z=df[mask]['z'],
            mode='markers',
            name=section,
            marker=dict(size=6),
            text=df[mask].apply(
                lambda x: f"Setting: {x['setting']}<br>Type: {x['component_type']}<br>Description: {x['description']}",
                axis=1
            ),
            hoverinfo='text'
        ))
    
    # Update layout for better visualization
    fig.update_layout(
        title='3D Settings Documentation Embeddings',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_dark"
    )
    
    # Add buttons for different views
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Reset View",
                        method="relayout",
                        args=[{"scene.camera": dict(
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=1.5, y=1.5, z=1.5)
                        )}]
                    ),
                    dict(
                        label="Top View",
                        method="relayout",
                        args=[{"scene.camera": dict(
                            up=dict(x=0, y=1, z=0),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=0, y=0, z=2)
                        )}]
                    ),
                    dict(
                        label="Side View",
                        method="relayout",
                        args=[{"scene.camera": dict(
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=2, y=0, z=0)
                        )}]
                    )
                ]
            )
        ]
    )
    
    # Save the enhanced 3D visualization
    output_path = "/app/visualizations/embeddings_3d.html"
    fig.write_html(output_path)
    print(f"3D visualization saved to {output_path}")
    
    # Save the data for further analysis
    df.to_csv("/app/visualizations/embeddings_metadata.csv", index=False)
    print("Metadata saved to /app/visualizations/embeddings_metadata.csv")
    
    return fig

# Create visualizations
# create_3d_visualization(pipeline)

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

