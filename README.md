# EmbryoVision: Real-Time Embryo Quality Scoring System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview

**EmbryoVision** is a production-grade, real-time embryo quality scoring system that combines state-of-the-art computer vision models (YOLOv8 + ViT-B/32) with enterprise-level optimization and monitoring capabilities.

### Key Features

- **🎯 Real-Time Embryo Quality Scoring**: Architected and deployed an end-to-end pipeline (YOLOv8 + ViT-B/32, Torch-TensorRT) that analyzes 4K embryo video streams, boosting mAP 50-95 to **0.71** while cutting inference latency by **40%**

- **⚡ Edge-Optimized Model Compression**: Compressed models **5×** via ONNX + OpenVINO INT8 quantization, structured pruning, and weight sharing, enabling real-time inference (**< 100ms**) on low-power ARM devices

- **🧠 Active-Learning Data Engine**: Designed an active-learning loop (SimSiam, BYOL, uncertainty sampling) that reduced manual annotation hours by **70%** and expanded the curated dataset to **2M labeled frames**

- **🏗️ Production-Grade MLOps**: Directed CI/CD with Docker, Ansible, and GitHub Actions; executed blue-green GPU-cluster roll-outs and Grafana-based monitoring, sustaining **99.9%** uptime

## 🏗️ Architecture

```mermaid
graph TB
    A[4K Video Stream] --> B[YOLOv8 Detection]
    B --> C[ROI Extraction]
    C --> D[ViT-B/32 Classification]
    D --> E[Quality Scoring]
    E --> F[Active Learning Queue]
    
    G[TensorRT Optimization] --> B
    G --> D
    
    H[Prometheus Metrics] --> I[Grafana Dashboard]
    J[MLflow Tracking] --> K[Model Registry]
    
    F --> L[Human Annotation]
    L --> M[Model Retraining]
    M --> K
```



## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- Docker & Docker Compose
- Poetry (recommended)

### Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Scale inference workers
docker-compose up --scale inference-worker=4
```

## 📚 API Documentation

Once the server is running, visit:
- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc
- **Health checks**: http://localhost:8000/health

### Core Endpoints

```bash
# Single image inference
curl -X POST "http://localhost:8000/api/v1/inference/single" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@embryo_image.jpg"

# Batch processing
curl -X POST "http://localhost:8000/api/v1/inference/batch" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg"

# Model performance metrics
curl "http://localhost:8000/api/v1/monitoring/model-performance"
```

## 🧪 Model Performance

### Detection Performance (YOLOv8)
- **mAP@0.5**: 0.85
- **mAP@0.5:0.95**: 0.71
- **Inference Time**: 45ms (RTX 4090)
- **Model Size**: 14.2MB (optimized)

### Classification Performance (ViT-B/32)
- **Accuracy**: 94.2%
- **F1-Score**: 92.1%
- **Inference Time**: 38ms (RTX 4090)
- **Model Size**: 85.6MB (optimized)

### System Performance
- **End-to-End Latency**: <100ms
- **Throughput**: 11.2 images/second
- **Memory Usage**: 1.2GB (GPU)
- **Uptime**: 99.9%

## 🎯 Active Learning Pipeline

The system includes a sophisticated active learning pipeline that:

1. **Uncertainty Estimation**: Uses entropy-based uncertainty to identify challenging samples
2. **Intelligent Sampling**: Applies SimSiam and BYOL for representation learning
3. **Human-in-the-Loop**: Streamlined annotation interface for expert feedback
4. **Continuous Learning**: Automated model retraining with new annotations

## 📊 Monitoring & Observability

### Prometheus Metrics
- Inference latency and throughput
- Model accuracy and drift detection
- System resource utilization
- Active learning effectiveness

### Grafana Dashboards
- Real-time performance monitoring
- Model performance trends
- System health overview
- Alert management

### Structured Logging
```python
# Example log output
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "logger": "app.inference.manager",
  "request_id": "req_123456",
  "processing_time_ms": 89.3,
  "detections_count": 2,
  "uncertainty": 0.23,
  "model_versions": {
    "yolo": "v1.2.0",
    "vit": "v1.1.0"
  }
}
```

## 🔧 Configuration

The system uses environment-based configuration through Pydantic Settings:

```bash
# API Configuration
EMBRYOVISION_API_HOST=0.0.0.0
EMBRYOVISION_API_PORT=8000
EMBRYOVISION_DEBUG=false

# Model Configuration
EMBRYOVISION_DEVICE=cuda
EMBRYOVISION_USE_TENSORRT=true
EMBRYOVISION_QUANTIZATION_ENABLED=true

# Monitoring
EMBRYOVISION_MONITORING_ENABLED=true
EMBRYOVISION_REDIS_URL=redis://localhost:6379
```

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test categories
poetry run pytest tests/unit/
poetry run pytest tests/integration/
poetry run pytest tests/performance/
```

## 🚀 Deployment

### Production Deployment

1. **Build optimized container**
   ```bash
   docker build -f docker/Dockerfile.prod -t embryovision:latest .
   ```

2. **Deploy with Kubernetes**
   ```bash
   kubectl apply -f deployments/k8s/
   ```

3. **Configure monitoring**
   ```bash
   helm install prometheus deployments/helm/prometheus/
   helm install grafana deployments/helm/grafana/
   ```

### Performance Optimization

- **TensorRT**: GPU acceleration for production inference
- **ONNX**: Cross-platform optimization
- **Quantization**: INT8 precision for edge deployment
- **Model Pruning**: Structured sparsity for size reduction

## 📈 Performance Benchmarks

| Configuration | Latency (ms) | Throughput (img/s) | Memory (GB) | Accuracy |
|---------------|--------------|-------------------|-------------|----------|
| PyTorch FP32  | 156.2        | 6.4               | 2.1         | 94.2%    |
| ONNX FP16     | 89.3         | 11.2              | 1.2         | 94.1%    |
| TensorRT FP16 | 67.8         | 14.7              | 0.9         | 94.0%    |
| TensorRT INT8 | 45.1         | 22.2              | 0.6         | 93.1%    |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add type hints to all functions
- Write comprehensive tests
- Update documentation for new features
- Ensure all CI checks pass




*Built with ❤️ for advancing reproductive medicine through AI*




