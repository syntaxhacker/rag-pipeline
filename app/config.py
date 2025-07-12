from typing import Dict, Optional, List
from dataclasses import dataclass
from haystack.dataclasses import ChatMessage

@dataclass
class DatasetConfig:
    name: str
    split: str = "train"
    content_field: str = "content"
    fields: Dict[str, str] = None  # Dictionary of field mappings
    prompt_template: Optional[str] = None

# Default configurations for different datasets
DATASET_CONFIGS = {
    "awesome-chatgpt-prompts": DatasetConfig(
        name="fka/awesome-chatgpt-prompts",
        content_field="prompt",
        fields={
            "role": "act",
            "prompt": "prompt"
        },
        prompt_template="""
        Given the following context where each document represents a prompt for a specific role,
        please answer the question while considering both the role and the prompt content.
        
        Available Contexts:
        {% for document in documents %}
            {% if document.meta.role %}Role: {{ document.meta.role }}{% endif %}
            Content: {{ document.content }}
            ---
        {% endfor %}
        
        Question: {{question}}
        Answer:
        """
    ),
    "settings-dataset": DatasetConfig(
        name="syntaxhacker/rag_pipeline",
        content_field="context",
        fields={
            "question": "question",
            "answer": "answer",
            "context": "context"
        },
        prompt_template="""
        Given the following context about software settings and configurations,
        please answer the question accurately based on the provided information.
        
        For each setting, provide a clear, step-by-step navigation path and include:
        1. The exact location (Origin Type > Tab > Section > Setting name)
        2. What the setting does
        3. Available options/values
        4. How to access and modify the setting
        5. Reference screenshots (if available)
        
        Format your answer as:
        "To [accomplish task], follow these steps:

        Location: [Origin Type] > [Tab] > [Section] > [Setting name]
        Purpose: [describe what the setting does]
        Options: [list available values/options]
        How to set: [describe interaction method: toggle/select/input]
        
        Visual Guide:
        [Include reference image links if available]

        For more details, you can refer to the screenshots above showing the exact location and interface."

        Available Contexts:
        {% for document in documents %}
            Setting Info: {{ document.content }}
            Reference Answer: {{ document.meta.answer }}
            ---
        {% endfor %}

        Question: {{question}}
        Answer:
        """
    ),
    "seven-wonders": DatasetConfig(
        name="bilgeyucel/seven-wonders",
        content_field="content",
        fields={},  # No additional fields needed
        prompt_template="""
        Given the following information about the Seven Wonders, please answer the question.
        
        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}
        
        Question: {{question}}
        Answer:
        """
    ),
    "psychology-dataset": DatasetConfig(
        name="jkhedri/psychology-dataset",
        split="train",
        content_field="question",  # Assuming we want to use the question as the content
        fields={
            "response_j": "response_j",  # Response from one model
            "response_k": "response_k"   # Response from another model
        },
        prompt_template="""
        Given the following context where each document represents a psychological inquiry,
        please answer the question based on the provided responses.

        Available Contexts:
        {% for document in documents %}
            Question: {{ document.content }}
            Response J: {{ document.meta.response_j }}
            Response K: {{ document.meta.response_k }}
            ---
        {% endfor %}

        Question: {{question}}
        Answer:
        """
    ),
    "developer-portfolio": DatasetConfig(
        name="syntaxhacker/developer-portfolio-rag",
        split="train",
        content_field="answer",
        fields={
            "question": "question",
            "answer": "answer",
            "context": "context"
        },
        prompt_template="""
        Given the following context about a software developer's skills, experience, and background,
        please answer the question accurately based on the provided information.
        
        For each query, provide detailed information about:
        1. Technical skills and programming languages
        2. Machine learning and AI experience
        3. Projects and professional experience
        4. Tools and frameworks used
        5. Personal interests and learning approach
        
        Available Contexts:
        {% for document in documents %}
            Question: {{ document.meta.question }}
            Answer: {{ document.content }}
            Context: {{ document.meta.context }}
            ---
        {% endfor %}
        
        Question: {{question}}
        Answer:
        """
    ),
}

# Default configuration for embedding and LLM models
MODEL_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "gemini-2.0-flash-exp",
} 