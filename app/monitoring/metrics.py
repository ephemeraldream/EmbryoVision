"""Metrics collection and monitoring system."""

import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque

import numpy as np
import redis
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

from ..core.logging import get_logger
from ..core.config import settings


logger = get_logger(__name__)


class MetricsCollector:
    """
    Production-grade metrics collection system.
    
    Features:
    - Prometheus metrics integration
    - Real-time performance tracking
    - Active learning metrics
    - Model performance monitoring
    - System resource tracking
    """
    
    def __init__(self):
        # Prometheus registry
        self.registry = CollectorRegistry()
        
        # Inference metrics
        self.inference_counter = Counter(
            'embryovision_inference_total',
            'Total inference requests',
            ['model_type', 'status'],
            registry=self.registry
        )
        
        self.inference_duration = Histogram(
            'embryovision_inference_duration_seconds',
            'Inference duration in seconds',
            ['model_type'],
            registry=self.registry
        )
        
        self.detection_count = Histogram(
            'embryovision_detections_per_image',
            'Number of detections per image',
            buckets=[0, 1, 2, 3, 5, 8, 13, float('inf')],
            registry=self.registry
        )
        
        self.uncertainty_gauge = Gauge(
            'embryovision_prediction_uncertainty',
            'Prediction uncertainty score',
            ['model_type'],
            registry=self.registry
        )
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            'embryovision_model_accuracy',
            'Model accuracy score',
            ['model_type', 'dataset'],
            registry=self.registry
        )
        
        self.model_latency = Gauge(
            'embryovision_model_latency_ms',
            'Model inference latency in milliseconds',
            ['model_type', 'optimization'],
            registry=self.registry
        )
        
        # Active learning metrics
        self.annotation_queue_size = Gauge(
            'embryovision_annotation_queue_size',
            'Size of annotation queue',
            registry=self.registry
        )
        
        self.feedback_counter = Counter(
            'embryovision_feedback_total',
            'Total feedback submissions',
            ['feedback_type'],
            registry=self.registry
        )
        
        # System metrics
        self.memory_usage = Gauge(
            'embryovision_memory_usage_bytes',
            'Memory usage in bytes',
            ['component'],
            registry=self.registry
        )
        
        self.gpu_utilization = Gauge(
            'embryovision_gpu_utilization_percent',
            'GPU utilization percentage',
            registry=self.registry
        )
        
        # Redis connection for distributed metrics
        self.redis_client: Optional[redis.Redis] = None
        
        # In-memory metrics storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.request_times = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        
        # Performance windows
        self.minute_windows = defaultdict(lambda: deque(maxlen=60))
        self.hour_windows = defaultdict(lambda: deque(maxlen=24))
        
    async def initialize(self) -> None:
        """Initialize metrics collection system."""
        logger.info("Initializing metrics collector")
        
        try:
            # Initialize Redis connection
            if settings.redis_url:
                self.redis_client = redis.from_url(settings.redis_url)
                await self._test_redis_connection()
            
            # Start background metrics collection
            asyncio.create_task(self._collect_system_metrics())
            asyncio.create_task(self._cleanup_old_metrics())
            
            logger.info("Metrics collector initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize metrics collector", error=str(e))
            # Continue without Redis if it fails
    
    async def record_inference(
        self, 
        request_id: str, 
        processing_time_ms: float, 
        detections_count: int, 
        uncertainty: float,
        model_type: str = "pipeline",
        status: str = "success"
    ) -> None:
        """Record inference metrics."""
        try:
            # Prometheus metrics
            self.inference_counter.labels(
                model_type=model_type, 
                status=status
            ).inc()
            
            self.inference_duration.labels(
                model_type=model_type
            ).observe(processing_time_ms / 1000.0)
            
            self.detection_count.observe(detections_count)
            
            self.uncertainty_gauge.labels(
                model_type=model_type
            ).set(uncertainty)
            
            # In-memory metrics
            current_time = time.time()
            self.request_times.append(current_time)
            
            metrics_data = {
                'request_id': request_id,
                'processing_time_ms': processing_time_ms,
                'detections_count': detections_count,
                'uncertainty': uncertainty,
                'timestamp': current_time,
                'model_type': model_type,
                'status': status
            }
            
            self.metrics_history['inference'].append(metrics_data)
            
            # Time-windowed metrics
            minute_key = int(current_time // 60)
            self.minute_windows[minute_key].append(metrics_data)
            
            hour_key = int(current_time // 3600)
            self.hour_windows[hour_key].append(metrics_data)
            
            # Store in Redis if available
            if self.redis_client:
                await self._store_metrics_redis(metrics_data)
            
            logger.debug("Inference metrics recorded", request_id=request_id)
            
        except Exception as e:
            logger.error("Failed to record inference metrics", error=str(e))
    
    async def record_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """Record active learning feedback metrics."""
        try:
            feedback_type = feedback_data.get('type', 'correction')
            
            self.feedback_counter.labels(
                feedback_type=feedback_type
            ).inc()
            
            # Store detailed feedback metrics
            feedback_metrics = {
                'request_id': feedback_data.get('request_id'),
                'feedback_type': feedback_type,
                'annotation_time': feedback_data.get('annotation_time', 0),
                'corrections_count': len(feedback_data.get('corrections', [])),
                'timestamp': time.time()
            }
            
            self.metrics_history['feedback'].append(feedback_metrics)
            
            logger.debug("Feedback metrics recorded", 
                        request_id=feedback_data.get('request_id'))
            
        except Exception as e:
            logger.error("Failed to record feedback metrics", error=str(e))
    
    async def record_model_performance(
        self, 
        model_type: str, 
        metrics: Dict[str, float]
    ) -> None:
        """Record model performance metrics."""
        try:
            # Update Prometheus gauges
            if 'accuracy' in metrics:
                self.model_accuracy.labels(
                    model_type=model_type,
                    dataset='validation'
                ).set(metrics['accuracy'])
            
            if 'avg_inference_time_ms' in metrics:
                self.model_latency.labels(
                    model_type=model_type,
                    optimization='current'
                ).set(metrics['avg_inference_time_ms'])
            
            # Store in history
            performance_data = {
                'model_type': model_type,
                'metrics': metrics,
                'timestamp': time.time()
            }
            
            self.metrics_history['model_performance'].append(performance_data)
            
            logger.debug("Model performance metrics recorded", model_type=model_type)
            
        except Exception as e:
            logger.error("Failed to record model performance", error=str(e))
    
    async def update_annotation_queue_size(self, size: int) -> None:
        """Update annotation queue size metric."""
        try:
            self.annotation_queue_size.set(size)
            
            self.metrics_history['annotation_queue'].append({
                'size': size,
                'timestamp': time.time()
            })
            
        except Exception as e:
            logger.error("Failed to update annotation queue metrics", error=str(e))
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics."""
        try:
            current_time = time.time()
            
            # Recent inference times (last 100 requests)
            recent_times = []
            for data in list(self.metrics_history['inference'])[-100:]:
                if current_time - data['timestamp'] < 300:  # Last 5 minutes
                    recent_times.append(data['processing_time_ms'])
            
            # Calculate statistics
            if recent_times:
                avg_time = np.mean(recent_times)
                p95_time = np.percentile(recent_times, 95)
                p99_time = np.percentile(recent_times, 99)
                throughput = len(recent_times) / 5.0 * 60  # requests per minute
            else:
                avg_time = p95_time = p99_time = throughput = 0
            
            # Error rate (last hour)
            hour_ago = current_time - 3600
            total_requests = 0
            error_requests = 0
            
            for data in self.metrics_history['inference']:
                if data['timestamp'] > hour_ago:
                    total_requests += 1
                    if data['status'] != 'success':
                        error_requests += 1
            
            error_rate = error_requests / max(total_requests, 1)
            
            # Queue metrics
            queue_sizes = [
                data['size'] for data in self.metrics_history['annotation_queue']
                if current_time - data['timestamp'] < 60
            ]
            current_queue_size = queue_sizes[-1] if queue_sizes else 0
            
            return {
                'inference': {
                    'avg_processing_time_ms': avg_time,
                    'p95_processing_time_ms': p95_time,
                    'p99_processing_time_ms': p99_time,
                    'throughput_per_minute': throughput,
                    'error_rate_percent': error_rate * 100,
                    'total_requests_last_hour': total_requests
                },
                'active_learning': {
                    'annotation_queue_size': current_queue_size,
                    'feedback_submissions_last_hour': len([
                        data for data in self.metrics_history['feedback']
                        if current_time - data['timestamp'] < 3600
                    ])
                },
                'timestamp': current_time
            }
            
        except Exception as e:
            logger.error("Failed to get real-time metrics", error=str(e))
            return {}
    
    def get_historical_metrics(
        self, 
        hours: int = 24
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get historical metrics for the specified time period."""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            historical_data = {}
            
            for metric_type, data_list in self.metrics_history.items():
                filtered_data = [
                    data for data in data_list
                    if data['timestamp'] > cutoff_time
                ]
                historical_data[metric_type] = filtered_data
            
            return historical_data
            
        except Exception as e:
            logger.error("Failed to get historical metrics", error=str(e))
            return {}
    
    async def _collect_system_metrics(self) -> None:
        """Background task to collect system metrics."""
        while True:
            try:
                import psutil
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.labels(component='system').set(memory.used)
                
                # GPU metrics
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_utilization.set(gpu_util.gpu)
                except ImportError:
                    pass  # pynvml not available
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logger.warning("System metrics collection failed", error=str(e))
                await asyncio.sleep(60)  # Retry after 1 minute
    
    async def _cleanup_old_metrics(self) -> None:
        """Background task to clean up old metrics."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                current_time = time.time()
                max_age = 24 * 3600  # Keep 24 hours of data
                
                # Clean up in-memory metrics
                for metric_type in self.metrics_history:
                    old_data = self.metrics_history[metric_type]
                    new_data = deque([
                        data for data in old_data
                        if current_time - data['timestamp'] < max_age
                    ], maxlen=1000)
                    self.metrics_history[metric_type] = new_data
                
                # Clean up time windows
                cutoff_minutes = int((current_time - max_age) // 60)
                for minute_key in list(self.minute_windows.keys()):
                    if minute_key < cutoff_minutes:
                        del self.minute_windows[minute_key]
                
                cutoff_hours = int((current_time - max_age) // 3600)
                for hour_key in list(self.hour_windows.keys()):
                    if hour_key < cutoff_hours:
                        del self.hour_windows[hour_key]
                
                logger.debug("Metrics cleanup completed")
                
            except Exception as e:
                logger.error("Metrics cleanup failed", error=str(e))
    
    async def _test_redis_connection(self) -> None:
        """Test Redis connection."""
        try:
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning("Redis connection failed", error=str(e))
            self.redis_client = None
    
    async def _store_metrics_redis(self, metrics_data: Dict[str, Any]) -> None:
        """Store metrics in Redis for distributed access."""
        try:
            if self.redis_client:
                key = f"embryovision:metrics:{int(time.time())}"
                self.redis_client.setex(
                    key, 
                    86400,  # 24 hours TTL
                    str(metrics_data)
                )
        except Exception as e:
            logger.warning("Redis metrics storage failed", error=str(e))
    
    async def cleanup(self) -> None:
        """Clean up metrics collector resources."""
        logger.info("Cleaning up metrics collector")
        
        if self.redis_client:
            self.redis_client.close()
    
    def get_prometheus_registry(self) -> CollectorRegistry:
        """Get Prometheus metrics registry."""
        return self.registry 