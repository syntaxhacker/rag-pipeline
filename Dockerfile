FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and clean up in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal requirements and install dependencies
COPY requirements-ultra-minimal.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-ultra-minimal.txt \
    && pip cache purge

# Copy application code
COPY app/ app/
COPY start.sh .

# Make start script executable
RUN chmod +x start.sh

# Expose the port
EXPOSE 8000

# Run the FastAPI application
CMD ["./start.sh"] 