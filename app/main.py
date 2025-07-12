from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
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

app = FastAPI()

# Initialize pipelines for all datasets
pipelines = {}
google_api_key = os.getenv("GOOGLE_API_KEY")

for dataset_name in DATASET_CONFIGS.keys():
    try:
        print(f"Loading dataset: {dataset_name}")
        pipeline = RAGPipeline.from_preset(
            google_api_key=google_api_key,
            preset_name=dataset_name
        )
        pipelines[dataset_name] = pipeline
        print(f"Successfully loaded {dataset_name}")
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")

# Use settings-dataset as default
default_pipeline = pipelines.get("settings-dataset")

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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "datasets_loaded": len(pipelines)}

