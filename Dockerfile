# Multi-stage build for AgentNet
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up build environment
WORKDIR /build
COPY pyproject.toml ./
COPY README.md ./
COPY agentnet/ ./agentnet/

# Install Python dependencies and build wheel
RUN pip install --no-cache-dir build wheel
RUN python -m build

# Production stage
FROM python:3.11-slim as production

# Add labels for container metadata
LABEL org.opencontainers.image.title="AgentNet"
LABEL org.opencontainers.image.description="A governed multi-agent reasoning platform"
LABEL org.opencontainers.image.version=${VERSION}
LABEL org.opencontainers.image.created=${BUILD_DATE}
LABEL org.opencontainers.image.revision=${VCS_REF}
LABEL org.opencontainers.image.source="https://github.com/V1B3hR/agentnet"
LABEL org.opencontainers.image.licenses="GPL-3.0"

# Create non-root user
RUN groupadd -r agentnet && useradd -r -g agentnet agentnet

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up application directory
WORKDIR /app
RUN chown -R agentnet:agentnet /app

# Copy built wheel from builder stage
COPY --from=builder /build/dist/*.whl ./

# Install AgentNet
RUN pip install --no-cache-dir *.whl[full] && rm *.whl

# Create directories for data persistence
RUN mkdir -p /app/data/cost_logs /app/data/sessions /app/data/risk_logs \
    && chown -R agentnet:agentnet /app/data

# Copy configuration files
COPY configs/ ./configs/
RUN chown -R agentnet:agentnet ./configs/

# Switch to non-root user
USER agentnet

# Set environment variables
ENV PYTHONPATH=/app
ENV AGENTNET_DATA_DIR=/app/data
ENV AGENTNET_CONFIG_DIR=/app/configs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import agentnet; print('Health check passed')" || exit 1

# Expose default port
EXPOSE 8000

# Default command
CMD ["python", "-m", "agentnet", "--host", "0.0.0.0", "--port", "8000"]