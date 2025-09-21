import asyncio
from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...core.logging import get_logger
from ...core.config import settings
from ...inference.manager import InferenceManager


logger = get_logger(__name__)
router = APIRouter()


class EmbryoDetection(BaseModel):
    bbox: List[float] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Detection confidence score")
    class_id: int = Field(..., description="Detected class ID")
    class_name: str = Field(..., description="Detected class name")


class QualityScore(BaseModel):
    overall_score: float = Field(..., description="Overall quality score (0-1)")
    morphology_score: float = Field(..., description="Morphological quality score")
    development_stage: str = Field(..., description="Development stage classification")
    viability_probability: float = Field(..., description="Predicted viability probability")
    uncertainty: float = Field(..., description="Model uncertainty score")


class InferenceResult(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    detections: List[EmbryoDetection] = Field(..., description="Detected embryos")
    quality_scores: List[QualityScore] = Field(..., description="Quality assessments")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    model_versions: Dict[str, str] = Field(..., description="Model version information")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")


class BatchInferenceRequest(BaseModel):
    images: List[str] = Field(..., description="Base64 encoded images")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Inference options")


class ActiveLearningFeedback(BaseModel):
    request_id: str = Field(..., description="Original request ID")
    corrections: List[Dict[str, Any]] = Field(..., description="Human corrections")
    annotation_time: float = Field(..., description="Time spent on annotation")


async def get_inference_manager() -> InferenceManager:
    from fastapi import Request
    return InferenceManager()


@router.post("/single", response_model=InferenceResult)
async def analyze_single_image(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    manager: InferenceManager = Depends(get_inference_manager)
) -> InferenceResult:
    import time
    start_time = time.time()
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if file.size > settings.upload_max_size:
        raise HTTPException(status_code=413, detail="File too large")
    
    request_id = str(uuid4())
    logger.info("Processing single image", request_id=request_id, filename=file.filename)
    
    try:
        image_data = await file.read()
        result = await manager.process_image(image_data, request_id)
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(
            "Inference completed",
            request_id=request_id,
            processing_time_ms=processing_time,
            detections_count=len(result.get("detections", [])),
        )
        
        if settings.active_learning_enabled and result.get("uncertainty", 0) > settings.uncertainty_threshold:
            background_tasks.add_task(manager.queue_for_annotation, request_id, image_data, result)
        
        return InferenceResult(
            request_id=request_id,
            detections=result["detections"],
            quality_scores=result["quality_scores"],
            processing_time_ms=processing_time,
            model_versions=result["model_versions"],
            metadata=result.get("metadata", {})
        )
        
    except Exception as e:
        logger.error("Inference failed", request_id=request_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.post("/batch", response_model=List[InferenceResult])
async def analyze_batch_images(
    files: List[UploadFile] = File(...),
    manager: InferenceManager = Depends(get_inference_manager)
) -> List[InferenceResult]:
    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size too large. Maximum: {settings.max_batch_size}"
        )
    
    logger.info("Processing batch", batch_size=len(files))
    
    try:
        tasks = []
        for file in files:
            image_data = await file.read()
            request_id = str(uuid4())
            task = manager.process_image(image_data, request_id)
            tasks.append((task, request_id))
        
        results = []
        for task, request_id in tasks:
            result = await task
            results.append(InferenceResult(
                request_id=request_id,
                detections=result["detections"],
                quality_scores=result["quality_scores"],
                processing_time_ms=result.get("processing_time_ms", 0),
                model_versions=result["model_versions"],
                metadata=result.get("metadata", {})
            ))
        
        logger.info("Batch processing completed", batch_size=len(results))
        return results
        
    except Exception as e:
        logger.error("Batch inference failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")


@router.post("/feedback")
async def submit_active_learning_feedback(
    feedback: ActiveLearningFeedback,
    manager: InferenceManager = Depends(get_inference_manager)
) -> JSONResponse:
    logger.info("Received active learning feedback", request_id=feedback.request_id)
    
    try:
        await manager.process_feedback(feedback.dict())
        return JSONResponse(
            status_code=200,
            content={"message": "Feedback received successfully"}
        )
    except Exception as e:
        logger.error("Feedback processing failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Feedback processing failed: {str(e)}")


@router.get("/models/status")
async def get_model_status(
    manager: InferenceManager = Depends(get_inference_manager)
) -> Dict[str, Any]:
    try:
        status = await manager.get_status()
        return status
    except Exception as e:
        logger.error("Failed to get model status", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")


@router.post("/models/reload")
async def reload_models(
    manager: InferenceManager = Depends(get_inference_manager)
) -> JSONResponse:
    try:
        await manager.reload_models()
        logger.info("Models reloaded successfully")
        return JSONResponse(
            status_code=200,
            content={"message": "Models reloaded successfully"}
        )
    except Exception as e:
        logger.error("Model reload failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}") 