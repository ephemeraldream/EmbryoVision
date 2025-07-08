from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from .core.config import settings
from .core.logging import configure_logging, get_logger, LoggingMiddleware
from .inference.manager import InferenceManager
from .api.routers import health, inference, monitoring
from .api.middleware import TimingMiddleware, ErrorHandlingMiddleware
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting EmbryoVision API", version=settings.api_version)
    
    app.state.inference_manager = InferenceManager()
    await app.state.inference_manager.initialize()
    
    if settings.monitoring_enabled:
        instrumentator = Instrumentator()
        instrumentator.instrument(app).expose(app)
        logger.info("Prometheus monitoring enabled")
    
    logger.info("Application startup complete")
    
    yield
    
    if hasattr(app.state, 'inference_manager'):
        await app.state.inference_manager.cleanup()
    
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        debug=settings.debug,
        lifespan=lifespan,
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(inference.router, prefix="/api/v1/inference", tags=["inference"])
    app.include_router(monitoring.router, prefix="/api/v1/monitoring", tags=["monitoring"])
    
    return app



app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_config=None,  # Use our custom logging
    ) 