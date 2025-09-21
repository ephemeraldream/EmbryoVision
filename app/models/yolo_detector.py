"""YOLOv8 detector for embryo detection with optimizations."""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import cv2
from ultralytics import YOLO
import onnxruntime as ort

from ..core.logging import get_logger
from ..core.config import settings


logger = get_logger(__name__)


class YOLODetector:
    """
    YOLOv8-based embryo detector with production optimizations.
    
    Features:
    - Real-time embryo detection
    - TensorRT optimization
    - ONNX runtime support
    - Batch processing
    - Model versioning
    """
    
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Model instances
        self.pytorch_model: Optional[YOLO] = None
        self.onnx_session: Optional[ort.InferenceSession] = None
        self.tensorrt_model: Optional[Any] = None
        
        # Model metadata
        self.model_version = "1.0.0"
        self.input_size = (640, 640)
        self.class_names = [
            "embryo_stage1", "embryo_stage2", "embryo_stage3", 
            "embryo_stage4", "embryo_stage5", "blastocyst"
        ]
        
        self.inference_times = []
        self.optimization_mode = "pytorch"  # pytorch, onnx, tensorrt
        
    async def initialize(self) -> None:
        """Initialize the YOLO model."""
        logger.info("Initializing YOLO detector", model_path=str(self.model_path))
        
        try:
            # Load PyTorch model
            if self.model_path.exists():
                self.pytorch_model = YOLO(str(self.model_path))
                self.pytorch_model.to(self.device)
                logger.info("PyTorch YOLO model loaded successfully")
            else:
                # Use pre-trained YOLOv8 and fine-tune for embryos
                self.pytorch_model = YOLO('yolov8n.pt')
                self.pytorch_model.to(self.device)
                logger.warning("Model file not found, using pre-trained YOLOv8")
            
            # Extract model metadata
            if hasattr(self.pytorch_model.model, 'yaml'):
                model_config = self.pytorch_model.model.yaml
                self.model_version = model_config.get('version', '1.0.0')
            
            logger.info("YOLO detector initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize YOLO detector", error=str(e), exc_info=True)
            raise
    
    async def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect embryos in the input image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            List of detection results with bounding boxes and metadata
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Run inference based on optimization mode
            if self.optimization_mode == "tensorrt" and self.tensorrt_model:
                detections = await self._detect_tensorrt(processed_image)
            elif self.optimization_mode == "onnx" and self.onnx_session:
                detections = await self._detect_onnx(processed_image)
            else:
                detections = await self._detect_pytorch(processed_image)
            
            # Post-process results
            results = self._postprocess_detections(detections, image.shape)
            
            # Track performance
            inference_time = (time.time() - start_time) * 1000
            self.inference_times.append(inference_time)
            
            logger.debug(
                "YOLO detection completed",
                detections_count=len(results),
                inference_time_ms=inference_time,
                optimization_mode=self.optimization_mode
            )
            
            return results
            
        except Exception as e:
            logger.error("YOLO detection failed", error=str(e), exc_info=True)
            raise
    
    async def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """Detect embryos in a batch of images for improved throughput."""
        if not images:
            return []
        
        try:
            # Process images concurrently
            tasks = [self.detect(image) for image in images]
            results = await asyncio.gather(*tasks)
            
            logger.info("Batch detection completed", batch_size=len(images))
            return results
            
        except Exception as e:
            logger.error("Batch detection failed", error=str(e), exc_info=True)
            raise
    
    async def _detect_pytorch(self, image: torch.Tensor) -> torch.Tensor:
        """Run inference using PyTorch model."""
        with torch.no_grad():
            results = self.pytorch_model(image, conf=self.confidence_threshold, iou=self.iou_threshold)
            return results[0].boxes if results[0].boxes is not None else torch.empty(0, 6)
    
    async def _detect_onnx(self, image: np.ndarray) -> np.ndarray:
        """Run inference using ONNX runtime."""
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: image})
        return outputs[0]
    
    async def _detect_tensorrt(self, image: np.ndarray) -> np.ndarray:
        """Run inference using TensorRT engine."""
        # TensorRT inference implementation
        # This would use the TensorRT Python API
        logger.warning("TensorRT inference not implemented yet")
        return await self._detect_pytorch(torch.from_numpy(image))
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for YOLO inference."""
        # Resize image to input size
        resized = cv2.resize(image, self.input_size)
        
        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0)
        
        # Move to device
        tensor = tensor.to(self.device)
        
        return tensor
    
    def _postprocess_detections(
        self, 
        detections: torch.Tensor, 
        original_shape: Tuple[int, int, int]
    ) -> List[Dict[str, Any]]:
        """Post-process YOLO detections."""
        results = []
        
        if len(detections) == 0:
            return results
        
        # Scale bounding boxes to original image size
        h_orig, w_orig = original_shape[:2]
        h_model, w_model = self.input_size
        
        scale_x = w_orig / w_model
        scale_y = h_orig / h_model
        
        for detection in detections:
            if isinstance(detection, torch.Tensor):
                detection = detection.cpu().numpy()
            
            # Extract coordinates and scores
            x1, y1, x2, y2 = detection[:4]
            confidence = detection[4] if len(detection) > 4 else 1.0
            class_id = int(detection[5]) if len(detection) > 5 else 0
            
            # Scale coordinates
            x1_scaled = x1 * scale_x
            y1_scaled = y1 * scale_y
            x2_scaled = x2 * scale_x
            y2_scaled = y2 * scale_y
            
            # Ensure coordinates are within image bounds
            x1_scaled = max(0, min(w_orig, x1_scaled))
            y1_scaled = max(0, min(h_orig, y1_scaled))
            x2_scaled = max(0, min(w_orig, x2_scaled))
            y2_scaled = max(0, min(h_orig, y2_scaled))
            
            # Skip invalid boxes
            if x2_scaled <= x1_scaled or y2_scaled <= y1_scaled:
                continue
            
            # Get class name
            class_name = (
                self.class_names[class_id] 
                if 0 <= class_id < len(self.class_names) 
                else f"class_{class_id}"
            )
            
            results.append({
                "bbox": [float(x1_scaled), float(y1_scaled), float(x2_scaled), float(y2_scaled)],
                "confidence": float(confidence),
                "class_id": int(class_id),
                "class_name": class_name,
                "area": float((x2_scaled - x1_scaled) * (y2_scaled - y1_scaled))
            })
        
        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return results
    
    def convert_to_onnx(self, output_path: str) -> None:
        """Convert PyTorch model to ONNX format."""
        logger.info("Converting YOLO model to ONNX", output_path=output_path)
        
        try:
            if self.pytorch_model is None:
                raise RuntimeError("PyTorch model not loaded")
            
            # Export to ONNX
            self.pytorch_model.export(format='onnx', imgsz=self.input_size)
            
            # Load ONNX session
            self.onnx_session = ort.InferenceSession(
                output_path,
                providers=[
                    'CUDAExecutionProvider' if self.device.type == 'cuda' else 'CPUExecutionProvider'
                ]
            )
            
            self.optimization_mode = "onnx"
            logger.info("ONNX conversion completed successfully")
            
        except Exception as e:
            logger.error("ONNX conversion failed", error=str(e), exc_info=True)
            raise
    
    def convert_to_tensorrt(self, output_path: str, batch_size: int = 1) -> None:
        """Convert model to TensorRT engine."""
        logger.info("Converting YOLO model to TensorRT", output_path=output_path)
        
        try:
            # This would implement TensorRT conversion
            # For now, just log the attempt
            logger.warning("TensorRT conversion not implemented yet")
            
        except Exception as e:
            logger.error("TensorRT conversion failed", error=str(e), exc_info=True)
            raise
    
    def get_version(self) -> str:
        """Get model version."""
        return self.model_version
    
    def get_optimizations(self) -> List[str]:
        """Get list of applied optimizations."""
        optimizations = []
        
        if self.onnx_session is not None:
            optimizations.append("ONNX")
        
        if self.tensorrt_model is not None:
            optimizations.append("TensorRT")
        
        if self.device.type == 'cuda':
            optimizations.append("CUDA")
        
        return optimizations
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        if not self.inference_times:
            return {}
        
        times = self.inference_times[-100:]  # Last 100 inferences
        
        return {
            "avg_inference_time_ms": np.mean(times),
            "p95_inference_time_ms": np.percentile(times, 95),
            "p99_inference_time_ms": np.percentile(times, 99),
            "min_inference_time_ms": np.min(times),
            "max_inference_time_ms": np.max(times)
        }
    
    async def reload(self) -> None:
        """Reload the model."""
        logger.info("Reloading YOLO detector")
        
        # Clear current models
        self.pytorch_model = None
        self.onnx_session = None
        self.tensorrt_model = None
        
        # Reinitialize
        await self.initialize()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up YOLO detector")
        
        self.pytorch_model = None
        
        if self.onnx_session:
            del self.onnx_session
            self.onnx_session = None
        
        if self.tensorrt_model:
            del self.tensorrt_model
            self.tensorrt_model = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 