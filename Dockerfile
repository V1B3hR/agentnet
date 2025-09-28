# AgentNet Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Install AgentNet in development mode
RUN pip install -e .

# Expose port for API server (if applicable)
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Health check
RUN python -c "import agentnet; print('AgentNet installed successfully')"

# Default command
CMD ["python", "-c", "import agentnet; print('AgentNet container ready')"]