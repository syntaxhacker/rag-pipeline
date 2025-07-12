from fastapi import FastAPI
from pydantic import BaseModel
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Pipeline API", description="Minimal RAG API", version="1.0.0")

class Question(BaseModel):
    text: str
    dataset: str = "developer-portfolio"

@app.get("/")
async def root():
    return {"status": "ok", "message": "RAG Pipeline API (Minimal Mode)", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "mode": "minimal", "port": os.getenv('PORT', '8000')}

@app.post("/answer")
async def get_answer(question: Question):
    # Static responses for developer portfolio questions
    responses = {
        "skills": "I'm a software developer with expertise in Python, JavaScript, React, FastAPI, machine learning, and cloud technologies. I enjoy building scalable web applications and working with AI/ML technologies.",
        "experience": "I have experience building full-stack applications, RESTful APIs, machine learning pipelines, and deploying applications to cloud platforms like Railway, Render, and Heroku.",
        "projects": "Some of my notable projects include RAG pipelines for Q&A systems, diffusion models for image generation, and various web applications using modern frameworks.",
        "hobbies": "I enjoy coding, learning new technologies, contributing to open source projects, and staying updated with the latest developments in AI and software engineering.",
        "contact": "You can reach me through my portfolio website or GitHub for collaboration opportunities.",
    }
    
    # Simple keyword matching
    text_lower = question.text.lower()
    for key, response in responses.items():
        if key in text_lower:
            return {"answer": response, "dataset": question.dataset, "mode": "minimal"}
    
    return {
        "answer": "I'm a passionate software developer specializing in Python, web development, and AI/ML. Feel free to ask about my skills, experience, projects, or hobbies!",
        "dataset": question.dataset,
        "mode": "minimal"
    }

@app.get("/datasets")
async def list_datasets():
    return {"datasets": ["developer-portfolio"], "mode": "minimal"}