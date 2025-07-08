# Multi-stage build for production optimization
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    curl \
    wget \
    git \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-glog0v5 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.7.1

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-dev --no-root && rm -rf $POETRY_CACHE_DIR

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Install the application
RUN poetry install --no-dev

# Create models and data directories
RUN mkdir -p /app/models /app/data /app/logs

# Production stage
FROM base as production

# Copy optimized models (if available)
COPY models/ ./models/

# Create non-root user for security
RUN groupadd -r embryovision && useradd -r -g embryovision embryovision
RUN chown -R embryovision:embryovision /app
USER embryovision

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM base as development

# Install development dependencies
RUN poetry install

# Install additional development tools
RUN pip install jupyter ipykernel

# Set development environment
ENV EMBRYOVISION_DEBUG=true

# Expose additional ports for development
EXPOSE 8000 8888

# Development command
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 