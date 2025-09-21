from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ...core.logging import get_logger
from ...core.config import settings


logger = get_logger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class DetailedHealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    components: dict


@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    from datetime import datetime
    
    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        timestamp=datetime.utcnow().isoformat()
    )


@router.get("/ready", response_model=DetailedHealthResponse)
async def readiness_check() -> DetailedHealthResponse:
    from datetime import datetime
    
    components = {
        "models": "healthy",
        "monitoring": "healthy" if settings.monitoring_enabled else "disabled",
        "logging": "healthy"
    }
    
    return DetailedHealthResponse(
        status="ready",
        version=settings.api_version,
        timestamp=datetime.utcnow().isoformat(),
        components=components
    )


@router.get("/live", response_model=HealthResponse)
async def liveness_check() -> HealthResponse:
    from datetime import datetime
    
    return HealthResponse(
        status="alive",
        version=settings.api_version,
        timestamp=datetime.utcnow().isoformat()
    ) 