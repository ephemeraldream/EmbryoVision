"""Inference manager for YOLOv8 + ViT-B/32 pipeline."""

import asyncio
import io
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json

import numpy as np
import torch
import cv2
from PIL import Image
import onnxruntime as ort
from ultralytics import YOLO

from ..core.config import settings
from ..core.logging import get_logger
from ..models.yolo_detector import YOLODetector
from ..models.vit_classifier import ViTClassifier
from ..models.optimization import ModelOptimizer
from ..monitoring.metrics import MetricsCollector


logger = get_logger(__name__)


class InferenceManager:
    """
    Manages the dual-network inference pipeline.
    
    Combines YOLOv8 for object detection with ViT-B/32 for quality classification,
    with optimizations including TensorRT, ONNX, and quantization.
    """
    
    def __init__(self):
        self.yolo_detector: Optional[YOLODetector] = None
        self.vit_classifier: Optional[ViTClassifier] = None
        self.optimizer: Optional[ModelOptimizer] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        
        self.device = torch.device(settings.device)
        self.initialized = False
        
        # Performance tracking
        self.inference_times = []
        self.request_count = 0
        self.error_count = 0
        
        # Active learning queue
        self.annotation_queue = asyncio.Queue(maxsize=settings.annotation_queue_size)
        
    async def initialize(self) -> None:
        """Initialize all models and optimization components."""
        logger.info("Initializing inference manager", device=settings.device)
        
        try:
            # Initialize metrics collector
            self.metrics_collector = MetricsCollector()
            
            # Initialize model optimizer
            self.optimizer = ModelOptimizer()
            
            # Initialize YOLO detector
            self.yolo_detector = YOLODetector(
                model_path=settings.yolo_model_path,
                device=self.device,
                confidence_threshold=settings.confidence_threshold,
                iou_threshold=settings.iou_threshold
            )
            await self.yolo_detector.initialize()
            
            # Initialize ViT classifier
            self.vit_classifier = ViTClassifier(
                model_path=settings.vit_model_path,
                device=self.device
            )
            await self.vit_classifier.initialize()
            
            # Apply optimizations
            if settings.use_tensorrt:
                await self._optimize_with_tensorrt()
            elif settings.use_onnx:
                await self._optimize_with_onnx()
            
            if settings.quantization_enabled:
                await self._apply_quantization()
            
            # Warm up models
            await self._warmup_models()
            
            self.initialized = True
            logger.info("Inference manager initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize inference manager", error=str(e), exc_info=True)
            raise
    
    async def process_image(self, image_data: bytes, request_id: str) -> Dict[str, Any]:
        """
        Process a single image through the dual-network pipeline.
        
        Args:
            image_data: Raw image bytes
            request_id: Unique request identifier
            
        Returns:
            Dictionary containing detection and classification results
        """
        if not self.initialized:
            raise RuntimeError("Inference manager not initialized")
        
        start_time = time.time()
        
        try:
            # Parse image
            image = self._parse_image(image_data)
            
            # Stage 1: YOLO detection
            detections = await self.yolo_detector.detect(image)
            logger.debug("YOLO detection completed", 
                        request_id=request_id, 
                        detections_count=len(detections))
            
            # Stage 2: ViT classification for each detection
            quality_scores = []
            for detection in detections:
                # Extract embryo region
                embryo_crop = self._extract_crop(image, detection['bbox'])
                
                # Classify quality
                quality_result = await self.vit_classifier.classify(embryo_crop)
                quality_scores.append(quality_result)
            
            # Calculate uncertainty for active learning
            uncertainty = self._calculate_uncertainty(quality_scores)
            
            # Prepare result
            result = {
                "detections": detections,
                "quality_scores": quality_scores,
                "uncertainty": uncertainty,
                "model_versions": {
                    "yolo": self.yolo_detector.get_version(),
                    "vit": self.vit_classifier.get_version()
                },
                "metadata": {
                    "image_size": image.shape[:2],
                    "processing_stages": ["detection", "classification"],
                    "optimization": {
                        "tensorrt_enabled": settings.use_tensorrt,
                        "onnx_enabled": settings.use_onnx,
                        "quantization_enabled": settings.quantization_enabled
                    }
                }
            }
            
            # Track metrics
            processing_time = (time.time() - start_time) * 1000
            self.inference_times.append(processing_time)
            self.request_count += 1
            
            if self.metrics_collector:
                await self.metrics_collector.record_inference(
                    request_id=request_id,
                    processing_time_ms=processing_time,
                    detections_count=len(detections),
                    uncertainty=uncertainty
                )
            
            result["processing_time_ms"] = processing_time
            
            logger.info("Image processing completed",
                       request_id=request_id,
                       processing_time_ms=processing_time,
                       detections_count=len(detections))
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error("Image processing failed", 
                        request_id=request_id, 
                        error=str(e), 
                        exc_info=True)
            raise
    
    async def process_batch(self, image_batch: List[bytes]) -> List[Dict[str, Any]]:
        """Process multiple images in batch for improved throughput."""
        if not self.initialized:
            raise RuntimeError("Inference manager not initialized")
        
        # Process images concurrently
        tasks = []
        for i, image_data in enumerate(image_batch):
            request_id = f"batch_{int(time.time())}_{i}"
            task = self.process_image(image_data, request_id)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch item {i} failed", error=str(result))
                processed_results.append({
                    "error": str(result),
                    "request_id": f"batch_{int(time.time())}_{i}"
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def queue_for_annotation(self, request_id: str, image_data: bytes, result: Dict[str, Any]) -> None:
        """Queue high-uncertainty samples for annotation."""
        if not settings.active_learning_enabled:
            return
        
        try:
            annotation_item = {
                "request_id": request_id,
                "image_data": image_data,
                "predictions": result,
                "timestamp": time.time(),
                "uncertainty": result.get("uncertainty", 0)
            }
            
            await self.annotation_queue.put(annotation_item)
            logger.info("Queued for annotation", request_id=request_id)
            
        except asyncio.QueueFull:
            logger.warning("Annotation queue full, dropping sample", request_id=request_id)
    
    async def process_feedback(self, feedback: Dict[str, Any]) -> None:
        """Process human feedback for active learning."""
        logger.info("Processing active learning feedback", 
                   request_id=feedback.get("request_id"))
        
        # In a real implementation, this would:
        # 1. Store the corrected annotations
        # 2. Trigger model retraining
        # 3. Update model weights
        # 4. Log improvement metrics
        
        if self.metrics_collector:
            await self.metrics_collector.record_feedback(feedback)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current status and performance metrics."""
        avg_inference_time = (
            sum(self.inference_times[-100:]) / len(self.inference_times[-100:])
            if self.inference_times else 0
        )
        
        return {
            "initialized": self.initialized,
            "device": str(self.device),
            "models": {
                "yolo": {
                    "loaded": self.yolo_detector is not None,
                    "version": self.yolo_detector.get_version() if self.yolo_detector else None,
                    "optimizations": self.yolo_detector.get_optimizations() if self.yolo_detector else []
                },
                "vit": {
                    "loaded": self.vit_classifier is not None,
                    "version": self.vit_classifier.get_version() if self.vit_classifier else None,
                    "optimizations": self.vit_classifier.get_optimizations() if self.vit_classifier else []
                }
            },
            "performance": {
                "total_requests": self.request_count,
                "total_errors": self.error_count,
                "error_rate": self.error_count / max(self.request_count, 1),
                "avg_inference_time_ms": avg_inference_time,
                "p95_inference_time_ms": np.percentile(self.inference_times[-100:], 95) if self.inference_times else 0
            },
            "active_learning": {
                "queue_size": self.annotation_queue.qsize(),
                "enabled": settings.active_learning_enabled
            }
        }
    
    async def reload_models(self) -> None:
        """Reload models (useful for A/B testing and updates)."""
        logger.info("Reloading models")
        
        # Reinitialize models
        if self.yolo_detector:
            await self.yolo_detector.reload()
        
        if self.vit_classifier:
            await self.vit_classifier.reload()
        
        # Reapply optimizations
        if settings.use_tensorrt:
            await self._optimize_with_tensorrt()
        elif settings.use_onnx:
            await self._optimize_with_onnx()
        
        # Warm up
        await self._warmup_models()
        
        logger.info("Models reloaded successfully")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up inference manager")
        
        if self.yolo_detector:
            await self.yolo_detector.cleanup()
        
        if self.vit_classifier:
            await self.vit_classifier.cleanup()
        
        if self.metrics_collector:
            await self.metrics_collector.cleanup()
    
    def _parse_image(self, image_data: bytes) -> np.ndarray:
        """Parse image data into numpy array."""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV compatibility
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            return image_array
            
        except Exception as e:
            logger.error("Failed to parse image", error=str(e))
            raise ValueError(f"Invalid image data: {e}")
    
    def _extract_crop(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Extract crop from image using bounding box."""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        crop = image[y1:y2, x1:x2]
        return crop
    
    def _calculate_uncertainty(self, quality_scores: List[Dict[str, Any]]) -> float:
        """Calculate uncertainty metric for active learning."""
        if not quality_scores:
            return 0.0
        
        # Simple uncertainty calculation based on prediction confidence
        uncertainties = []
        for score in quality_scores:
            # Use entropy or confidence-based uncertainty
            confidence = score.get('confidence', 0.5)
            uncertainty = 1.0 - confidence
            uncertainties.append(uncertainty)
        
        return np.mean(uncertainties) if uncertainties else 0.0
    
    async def _optimize_with_tensorrt(self) -> None:
        """Apply TensorRT optimization."""
        logger.info("Applying TensorRT optimization")
        
        if self.optimizer and self.yolo_detector:
            await self.optimizer.convert_to_tensorrt(
                self.yolo_detector,
                batch_size=settings.max_batch_size
            )
        
        if self.optimizer and self.vit_classifier:
            await self.optimizer.convert_to_tensorrt(
                self.vit_classifier,
                batch_size=settings.max_batch_size
            )
    
    async def _optimize_with_onnx(self) -> None:
        """Apply ONNX optimization."""
        logger.info("Applying ONNX optimization")
        
        if self.optimizer and self.yolo_detector:
            await self.optimizer.convert_to_onnx(self.yolo_detector)
        
        if self.optimizer and self.vit_classifier:
            await self.optimizer.convert_to_onnx(self.vit_classifier)
    
    async def _apply_quantization(self) -> None:
        """Apply INT8 quantization."""
        logger.info("Applying INT8 quantization")
        
        if self.optimizer:
            await self.optimizer.apply_quantization([
                self.yolo_detector,
                self.vit_classifier
            ])
    
    async def _warmup_models(self) -> None:
        """Warm up models with dummy inputs."""
        logger.info("Warming up models")
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Warmup YOLO
        if self.yolo_detector:
            await self.yolo_detector.detect(dummy_image)
        
        # Warmup ViT with dummy crop
        if self.vit_classifier:
            dummy_crop = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            await self.vit_classifier.classify(dummy_crop)
        
        logger.info("Model warmup completed") 