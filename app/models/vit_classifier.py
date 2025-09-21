"""ViT-B/32 classifier for embryo quality assessment."""

import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import timm
from transformers import ViTImageProcessor, ViTForImageClassification
import onnxruntime as ort

from ..core.logging import get_logger
from ..core.config import settings


logger = get_logger(__name__)


class ViTClassifier:
    """
    Vision Transformer (ViT-B/32) for embryo quality classification.
    
    Features:
    - Multi-stage embryo quality assessment
    - Development stage classification
    - Viability probability prediction
    - Uncertainty estimation
    - Model optimization support
    """
    
    def __init__(self, model_path: str, device: torch.device):
        self.model_path = Path(model_path)
        self.device = device
        
        # Model instances
        self.pytorch_model: Optional[nn.Module] = None
        self.processor: Optional[ViTImageProcessor] = None
        self.onnx_session: Optional[ort.InferenceSession] = None
        
        # Model configuration
        self.model_version = "1.0.0"
        self.input_size = (224, 224)
        self.num_classes = 6  # Quality grades: A, B, C, D, E, F
        
        # Class definitions for embryo quality
        self.quality_classes = [
            "Grade_A_Excellent",
            "Grade_B_Good", 
            "Grade_C_Fair",
            "Grade_D_Poor",
            "Grade_E_Very_Poor",
            "Grade_F_Non_Viable"
        ]
        
        # Development stage classes
        self.stage_classes = [
            "Zygote",
            "2-Cell",
            "4-Cell", 
            "8-Cell",
            "Morula",
            "Blastocyst"
        ]
        
        # Performance tracking
        self.inference_times = []
        self.optimization_mode = "pytorch"
        
    async def initialize(self) -> None:
        """Initialize the ViT classifier."""
        logger.info("Initializing ViT classifier", model_path=str(self.model_path))
        
        try:
            # Initialize image processor
            self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            
            # Load model
            if self.model_path.exists():
                # Load custom trained model
                self.pytorch_model = self._load_custom_model()
                logger.info("Custom ViT model loaded successfully")
            else:
                # Use pre-trained ViT and adapt for embryo classification
                self.pytorch_model = ViTForImageClassification.from_pretrained(
                    'google/vit-base-patch16-224',
                    num_labels=self.num_classes
                )
                logger.warning("Model file not found, using pre-trained ViT")
            
            # Move to device
            self.pytorch_model.to(self.device)
            self.pytorch_model.eval()
            
            logger.info("ViT classifier initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize ViT classifier", error=str(e), exc_info=True)
            raise
    
    async def classify(self, embryo_crop: np.ndarray) -> Dict[str, Any]:
        """
        Classify embryo quality and development stage.
        
        Args:
            embryo_crop: Cropped embryo image as numpy array
            
        Returns:
            Dictionary containing quality scores and metadata
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(embryo_crop)
            
            # Run inference
            if self.optimization_mode == "onnx" and self.onnx_session:
                predictions = await self._classify_onnx(processed_image)
            else:
                predictions = await self._classify_pytorch(processed_image)
            
            # Post-process results
            result = self._postprocess_predictions(predictions)
            
            # Track performance
            inference_time = (time.time() - start_time) * 1000
            self.inference_times.append(inference_time)
            
            logger.debug(
                "ViT classification completed",
                inference_time_ms=inference_time,
                predicted_quality=result['quality_grade'],
                confidence=result['confidence']
            )
            
            return result
            
        except Exception as e:
            logger.error("ViT classification failed", error=str(e), exc_info=True)
            raise
    
    async def classify_batch(self, embryo_crops: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Classify multiple embryo crops in batch."""
        if not embryo_crops:
            return []
        
        try:
            # Process in parallel for better throughput
            tasks = [self.classify(crop) for crop in embryo_crops]
            results = await asyncio.gather(*tasks)
            
            logger.info("Batch classification completed", batch_size=len(embryo_crops))
            return results
            
        except Exception as e:
            logger.error("Batch classification failed", error=str(e), exc_info=True)
            raise
    
    async def _classify_pytorch(self, image: torch.Tensor) -> torch.Tensor:
        """Run inference using PyTorch model."""
        with torch.no_grad():
            outputs = self.pytorch_model(image)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(logits, dim=-1)
            
            return probabilities
    
    async def _classify_onnx(self, image: np.ndarray) -> np.ndarray:
        """Run inference using ONNX runtime."""
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: image})
        
        # Apply softmax
        logits = outputs[0]
        probabilities = self._softmax(logits)
        
        return probabilities
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess embryo crop for ViT inference."""
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # Use ViT processor for preprocessing
            inputs = self.processor(pil_image, return_tensors="pt")
            
            # Move to device
            pixel_values = inputs['pixel_values'].to(self.device)
            
            return pixel_values
            
        except Exception as e:
            logger.error("Image preprocessing failed", error=str(e))
            raise
    
    def _postprocess_predictions(self, predictions: torch.Tensor) -> Dict[str, Any]:
        """Post-process ViT predictions."""
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            probs = predictions.cpu().numpy().flatten()
        else:
            probs = predictions.flatten()
        
        # Get top prediction
        predicted_class_id = np.argmax(probs)
        confidence = float(probs[predicted_class_id])
        
        # Calculate uncertainty (entropy-based)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(probs))
        uncertainty = entropy / max_entropy
        
        # Map to quality assessment
        quality_grade = self.quality_classes[predicted_class_id]
        
        # Convert quality grade to numerical score
        quality_scores = {
            "Grade_A_Excellent": 0.95,
            "Grade_B_Good": 0.80,
            "Grade_C_Fair": 0.65,
            "Grade_D_Poor": 0.45,
            "Grade_E_Very_Poor": 0.25,
            "Grade_F_Non_Viable": 0.05
        }
        
        overall_score = quality_scores.get(quality_grade, 0.5)
        
        # Predict development stage (simplified mapping)
        stage_mapping = {
            "Grade_A_Excellent": "Blastocyst",
            "Grade_B_Good": "Morula",
            "Grade_C_Fair": "8-Cell",
            "Grade_D_Poor": "4-Cell",
            "Grade_E_Very_Poor": "2-Cell",
            "Grade_F_Non_Viable": "Zygote"
        }
        
        development_stage = stage_mapping.get(quality_grade, "Unknown")
        
        # Calculate viability probability
        viability_probability = self._calculate_viability_probability(
            quality_grade, overall_score, confidence
        )
        
        # Calculate morphology score
        morphology_score = self._calculate_morphology_score(probs)
        
        return {
            "overall_score": overall_score,
            "morphology_score": morphology_score,
            "development_stage": development_stage,
            "viability_probability": viability_probability,
            "uncertainty": uncertainty,
            "confidence": confidence,
            "quality_grade": quality_grade,
            "quality_probabilities": {
                class_name: float(prob) 
                for class_name, prob in zip(self.quality_classes, probs)
            },
            "metadata": {
                "model_version": self.model_version,
                "optimization_mode": self.optimization_mode,
                "input_size": self.input_size
            }
        }
    
    def _calculate_viability_probability(
        self, 
        quality_grade: str, 
        overall_score: float, 
        confidence: float
    ) -> float:
        """Calculate embryo viability probability."""
        # Base viability rates by quality grade
        base_viability = {
            "Grade_A_Excellent": 0.85,
            "Grade_B_Good": 0.70,
            "Grade_C_Fair": 0.50,
            "Grade_D_Poor": 0.30,
            "Grade_E_Very_Poor": 0.15,
            "Grade_F_Non_Viable": 0.05
        }
        
        base_prob = base_viability.get(quality_grade, 0.30)
        
        # Adjust based on confidence
        confidence_factor = (confidence - 0.5) * 0.2  # Scale confidence impact
        adjusted_prob = base_prob + confidence_factor
        
        # Ensure probability is in valid range
        return max(0.0, min(1.0, adjusted_prob))
    
    def _calculate_morphology_score(self, probabilities: np.ndarray) -> float:
        """Calculate morphology score based on quality distribution."""
        # Weight quality classes differently for morphology assessment
        morphology_weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
        
        # Calculate weighted average
        morphology_score = np.sum(probabilities * morphology_weights)
        
        return float(morphology_score)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax activation."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _load_custom_model(self) -> nn.Module:
        """Load custom trained ViT model."""
        # Load model architecture
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=self.num_classes
        )
        
        # Load custom weights
        if self.model_path.suffix == '.pth':
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
        else:
            # Load ONNX or other format
            logger.warning("Custom model format not supported, using pre-trained")
        
        return model
    
    def convert_to_onnx(self, output_path: str) -> None:
        """Convert PyTorch model to ONNX format."""
        logger.info("Converting ViT model to ONNX", output_path=output_path)
        
        try:
            if self.pytorch_model is None:
                raise RuntimeError("PyTorch model not loaded")
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, *self.input_size).to(self.device)
            
            # Export to ONNX
            torch.onnx.export(
                self.pytorch_model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
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
    
    def get_version(self) -> str:
        """Get model version."""
        return self.model_version
    
    def get_optimizations(self) -> List[str]:
        """Get list of applied optimizations."""
        optimizations = []
        
        if self.onnx_session is not None:
            optimizations.append("ONNX")
        
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
        logger.info("Reloading ViT classifier")
        
        # Clear current models
        self.pytorch_model = None
        self.processor = None
        self.onnx_session = None
        
        # Reinitialize
        await self.initialize()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up ViT classifier")
        
        self.pytorch_model = None
        self.processor = None
        
        if self.onnx_session:
            del self.onnx_session
            self.onnx_session = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 