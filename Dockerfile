FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY start.sh .

# Make start script executable
RUN chmod +x start.sh

# Expose the port
EXPOSE 8000

# Run the FastAPI application
CMD ["./start.sh"] 