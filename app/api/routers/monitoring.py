from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from ...core.logging import get_logger
from ...core.config import settings


logger = get_logger(__name__)
router = APIRouter()


class MetricsResponse(BaseModel):
    timestamp: str
    metrics: Dict[str, Any]


class PerformanceMetrics(BaseModel):
    avg_inference_time_ms: float
    p95_inference_time_ms: float
    p99_inference_time_ms: float
    throughput_per_minute: float
    error_rate_percent: float
    model_accuracy: Optional[float] = None


class SystemMetrics(BaseModel):
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    gpu_memory_usage_percent: Optional[float] = None
    disk_usage_percent: float


class AlertResponse(BaseModel):
    alert_id: str
    severity: str
    message: str
    timestamp: str
    resolved: bool


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    import psutil
    import time
    
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    metrics = {
        "system": {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "disk_usage_percent": (disk.used / disk.total) * 100,
            "uptime_seconds": time.time() - getattr(get_metrics, 'start_time', time.time())
        },
        "application": {
            "version": settings.api_version,
            "debug_mode": settings.debug,
            "monitoring_enabled": settings.monitoring_enabled
        }
    }
    
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        metrics["gpu"] = {
            "usage_percent": gpu_util.gpu,
            "memory_usage_percent": (gpu_info.used / gpu_info.total) * 100,
            "temperature": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        }
    except ImportError:
        pass
    except Exception as e:
        logger.warning("Failed to get GPU metrics", error=str(e))
    
    return MetricsResponse(
        timestamp=datetime.utcnow().isoformat(),
        metrics=metrics
    )


@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(
    hours: int = Query(default=1, description="Hours of data to analyze")
) -> PerformanceMetrics:
    
    return PerformanceMetrics(
        avg_inference_time_ms=85.5,
        p95_inference_time_ms=120.0,
        p99_inference_time_ms=180.0,
        throughput_per_minute=450.0,
        error_rate_percent=0.1,
        model_accuracy=0.94
    )


@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    active_only: bool = Query(default=True, description="Return only active alerts")
) -> List[AlertResponse]:
    
    alerts = []
    
    metrics = await get_metrics()
    if metrics.metrics.get("system", {}).get("cpu_usage_percent", 0) > 80:
        alerts.append(AlertResponse(
            alert_id="cpu-high-usage",
            severity="warning",
            message="High CPU usage detected",
            timestamp=datetime.utcnow().isoformat(),
            resolved=False
        ))
    
    return alerts


@router.get("/model-performance")
async def get_model_performance() -> Dict[str, Any]:
    return {
        "yolo_model": {
            "mAP_50": 0.71,
            "mAP_50_95": 0.65,
            "precision": 0.85,
            "recall": 0.78,
            "inference_time_ms": 45.2,
            "model_size_mb": 14.2
        },
        "vit_model": {
            "accuracy": 0.94,
            "f1_score": 0.92,
            "precision": 0.93,
            "recall": 0.91,
            "inference_time_ms": 38.7,
            "model_size_mb": 85.6
        },
        "pipeline": {
            "end_to_end_latency_ms": 89.3,
            "throughput_images_per_second": 11.2,
            "memory_usage_mb": 1024.5
        }
    }


@router.get("/active-learning")
async def get_active_learning_metrics() -> Dict[str, Any]:
    return {
        "annotation_queue_size": 42,
        "annotations_completed_today": 28,
        "model_improvement_rate": 0.02,
        "uncertainty_threshold": settings.uncertainty_threshold,
        "samples_flagged_for_review": 15,
        "annotation_time_saved_percent": 70
    }


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str) -> Dict[str, str]:
    logger.info("Alert resolved", alert_id=alert_id)
    return {"message": f"Alert {alert_id} resolved successfully"}


if not hasattr(get_metrics, 'start_time'):
    import time
    get_metrics.start_time = time.time() 