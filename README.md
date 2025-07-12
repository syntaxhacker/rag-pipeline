# RAG Pipeline API

A multi-dataset Retrieval-Augmented Generation (RAG) pipeline built with FastAPI, Haystack, and Google's Gemini AI. This API supports multiple datasets and can answer questions using context-aware responses.

## Features

- **Multiple Dataset Support**: Load and query different datasets simultaneously
- **Developer Portfolio Q&A**: Custom dataset for answering questions about developer skills and experience
- **Settings Documentation**: Query software configuration settings and documentation
- **Psychology Dataset**: Access psychology-related Q&A content
- **Seven Wonders**: Information about the Seven Wonders of the World
- **ChatGPT Prompts**: Collection of awesome ChatGPT prompts
- **RESTful API**: Easy-to-use endpoints for querying datasets
- **Docker Support**: Containerized deployment
- **Cloud Deployment**: Ready for deployment on Render, Railway, or other platforms

## Quick Start

### Local Development

1. **Clone and navigate to the project**:
   ```bash
   git clone <your-repo-url>
   cd rag_pipeline
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**:
   ```bash
   export GOOGLE_API_KEY="your-google-api-key"
   export HF_TOKEN="your-huggingface-token"  # Optional, for dataset uploads
   ```

5. **Run the application**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Access the API**:
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs

### Docker Deployment

1. **Build the image**:
   ```bash
   docker build -t rag-api .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 \
     -e GOOGLE_API_KEY="your-google-api-key" \
     rag-api
   ```

## API Endpoints

### Health Check
```http
GET /health
```
Returns API health status and number of loaded datasets.

### List Datasets
```http
GET /datasets
```
Returns a list of all available datasets.

**Response:**
```json
{
  "datasets": [
    "awesome-chatgpt-prompts",
    "settings-dataset", 
    "seven-wonders",
    "psychology-dataset",
    "developer-portfolio"
  ]
}
```

### Ask Questions
```http
POST /answer
```

**Request Body:**
```json
{
  "text": "What programming languages do you know?",
  "dataset": "developer-portfolio"
}
```

**Response:**
```json
{
  "answer": "I specialize in Python, JavaScript/TypeScript, and have experience with machine learning frameworks like PyTorch and TensorFlow...",
  "dataset": "developer-portfolio"
}
```

## Available Datasets

### Developer Portfolio (`developer-portfolio`)
Custom Q&A dataset covering:
- Programming skills (Python, JavaScript, TypeScript)
- Machine learning experience (PyTorch, TensorFlow, Diffusion models)
- Tools and frameworks (React, Node.js, FastAPI, Docker)
- Projects and professional experience
- Personal interests and learning approach

**Example Questions:**
- "What programming languages do you know?"
- "What machine learning experience do you have?"
- "What are your hobbies and interests?"
- "What frameworks and tools do you use?"

### Settings Dataset (`settings-dataset`)
Software configuration and settings documentation.

**Example Questions:**
- "How do I configure delivery time settings?"
- "What is the grace period setting?"
- "How to enable geo fence exceptions?"

### Other Datasets
- `awesome-chatgpt-prompts`: Collection of ChatGPT prompts
- `seven-wonders`: Information about the Seven Wonders
- `psychology-dataset`: Psychology-related Q&A content

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google AI API key for Gemini | Yes |
| `HF_TOKEN` | Hugging Face token for dataset uploads | No |
| `PORT` | Port to run the application (default: 8000) | No |

### Dataset Configuration

Datasets are configured in `app/config.py`. Each dataset includes:
- **Name**: Hugging Face dataset identifier
- **Content Field**: Primary content field for retrieval
- **Fields**: Metadata field mappings
- **Prompt Template**: Custom prompt for the dataset

Example configuration:
```python
"developer-portfolio": DatasetConfig(
    name="syntaxhacker/developer-portfolio-rag",
    split="train",
    content_field="answer",
    fields={
        "question": "question",
        "answer": "answer", 
        "context": "context"
    },
    prompt_template="..."
)
```

## Deployment

### Render (Recommended)

1. **Push code to GitHub**
2. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Create new Web Service
   - Connect your GitHub repository
3. **Configure**:
   - Environment: Docker
   - Root Directory: `rag_pipeline`
   - Add environment variable: `GOOGLE_API_KEY`
4. **Deploy**: Automatic deployment from GitHub

### Other Platforms

The application is compatible with:
- **Railway**: Direct Docker deployment
- **Fly.io**: Use `flyctl` for deployment
- **Hugging Face Spaces**: Upload as Docker Space
- **AWS/GCP/Azure**: Container deployment

## Creating Custom Datasets

### 1. Prepare Your Data

Create a JSON file with Q&A pairs:
```json
[
  {
    "question": "Your question here",
    "answer": "Your answer here", 
    "context": {"category": "Technical Skills"}
  }
]
```

### 2. Upload to Hugging Face

Use the provided upload script:
```bash
# Set your HF token
export HF_TOKEN="your-token"

# Run upload script
python make_dataset/upload_developer_portfolio.py
```

### 3. Update Configuration

Add your dataset to `app/config.py`:
```python
"your-dataset": DatasetConfig(
    name="your-username/your-dataset-name",
    split="train",
    content_field="answer",
    fields={"question": "question", "answer": "answer"},
    prompt_template="Your custom prompt template..."
)
```

## Development

### Project Structure
```
rag_pipeline/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── pipeline.py      # RAG pipeline implementation
│   └── config.py        # Dataset configurations
├── make_dataset/
│   ├── *.json           # Dataset files
│   ├── *.parquet        # Processed datasets
│   └── upload_*.py      # Upload scripts
├── data/                # Local data directory
├── Dockerfile           # Container configuration
├── requirements.txt     # Python dependencies
├── render.yaml         # Render deployment config
└── README.md           # This file
```

### Adding New Features

1. **New Datasets**: Add configuration in `config.py`
2. **New Endpoints**: Add routes in `main.py`
3. **Custom Processing**: Modify `pipeline.py`

### Testing

```bash
# Install test dependencies
pip install pytest httpx

# Run tests
pytest tests/

# Test specific endpoint
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{"text": "test question", "dataset": "developer-portfolio"}'
```

## Troubleshooting

### Common Issues

1. **Port Binding Error**: Ensure `PORT` environment variable is set for cloud deployment
2. **Dataset Loading Failed**: Check Hugging Face token and dataset names
3. **Memory Issues**: Reduce number of datasets or upgrade deployment plan
4. **API Key Errors**: Verify `GOOGLE_API_KEY` is correctly set

### Logs

Check application logs for detailed error information:
- **Local**: Console output
- **Render**: Deployment logs in dashboard
- **Docker**: `docker logs <container-id>`

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Commit: `git commit -m "Add feature"`
6. Push: `git push origin feature-name`
7. Create a Pull Request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review Render deployment logs for cloud issues

---

**Live Demo**: [https://test-rag-myz2.onrender.com](https://test-rag-myz2.onrender.com)

**Documentation**: [FastAPI Interactive Docs](https://test-rag-myz2.onrender.com/docs)