import os
from typing import List, Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    
    api_title: str = "EmbryoVision API"
    api_description: str = "Real-Time Embryo Quality Scoring with YOLOv8 + ViT-B/32"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    yolo_model_path: str = Field(default="models/yolo/best.pt")
    vit_model_path: str = Field(default="models/vit/model.onnx")
    tensorrt_model_path: str = Field(default="models/tensorrt/model.engine")
    device: str = Field(default="cuda" if os.path.exists("/usr/local/cuda") else "cpu")
    
    max_batch_size: int = 32
    inference_timeout: int = 30
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    
    use_tensorrt: bool = True
    use_openvino: bool = False
    use_onnx: bool = True
    quantization_enabled: bool = True
    
    upload_max_size: int = 50 * 1024 * 1024
    allowed_extensions: List[str] = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
    data_dir: str = "data"
    
    uncertainty_threshold: float = 0.7
    active_learning_enabled: bool = True
    annotation_queue_size: int = 1000
    
    monitoring_enabled: bool = True
    prometheus_port: int = 9090
    grafana_enabled: bool = True
    
    log_level: str = "INFO"
    log_format: str = "json"
    
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "embryovision"
    
    wandb_project: str = "embryovision"
    wandb_entity: str = "embryovision-team"
    
    model_config = {
        "env_file": ".env",
        "env_prefix": "EMBRYOVISION_"
    }


settings = Settings() 