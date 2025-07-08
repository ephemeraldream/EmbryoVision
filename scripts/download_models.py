#!/usr/bin/env python3
"""Script to download pre-trained models for EmbryoVision."""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List
import urllib.request
import zipfile
import hashlib

import requests
from tqdm import tqdm


def calculate_md5(file_path: Path) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_file(url: str, destination: Path, expected_md5: str = None) -> bool:
    """Download a file with progress bar and verification."""
    print(f"Downloading {destination.name}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=destination.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        # Verify MD5 if provided
        if expected_md5:
            actual_md5 = calculate_md5(destination)
            if actual_md5 != expected_md5:
                print(f"MD5 verification failed for {destination.name}")
                destination.unlink()
                return False
            print(f"MD5 verification passed for {destination.name}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading {destination.name}: {e}")
        if destination.exists():
            destination.unlink()
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """Extract zip archive."""
    try:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        archive_path.unlink()  # Remove archive after extraction
        return True
    except Exception as e:
        print(f"Error extracting {archive_path}: {e}")
        return False


# Model definitions
MODELS = {
    "yolov8_embryo": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt",
        "destination": "models/yolo/yolov8n.pt",
        "md5": None,  # Will be updated with actual model
        "description": "YOLOv8 nano model (will be fine-tuned for embryos)"
    },
    "vit_base": {
        "url": "https://huggingface.co/google/vit-base-patch16-224/resolve/main/pytorch_model.bin",
        "destination": "models/vit/pytorch_model.bin",
        "md5": None,
        "description": "ViT-B/16 base model for classification"
    },
    "sample_images": {
        "url": "https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg",
        "destination": "data/samples/bus.jpg",
        "md5": None,
        "description": "Sample image for testing"
    }
}

# Custom models (if available)
CUSTOM_MODELS = {
    "embryo_yolo_trained": {
        "url": "https://your-storage.com/models/embryo_yolo_v1.pt",
        "destination": "models/yolo/best.pt",
        "md5": "your-model-md5-hash",
        "description": "Custom trained YOLOv8 for embryo detection"
    },
    "embryo_vit_trained": {
        "url": "https://your-storage.com/models/embryo_vit_v1.onnx",
        "destination": "models/vit/model.onnx",
        "md5": "your-model-md5-hash",
        "description": "Custom trained ViT for embryo classification"
    }
}


def create_directories():
    """Create necessary directories."""
    directories = [
        "models/yolo",
        "models/vit", 
        "models/onnx",
        "models/tensorrt",
        "models/optimized",
        "data/samples",
        "data/annotations",
        "data/training",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def download_base_models():
    """Download base pre-trained models."""
    print("Downloading base models...")
    
    success_count = 0
    total_count = len(MODELS)
    
    for model_name, model_info in MODELS.items():
        destination = Path(model_info["destination"])
        
        if destination.exists():
            print(f"Model {model_name} already exists, skipping...")
            success_count += 1
            continue
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        if download_file(
            model_info["url"], 
            destination, 
            model_info.get("md5")
        ):
            print(f"✓ Successfully downloaded {model_name}")
            success_count += 1
        else:
            print(f"✗ Failed to download {model_name}")
    
    print(f"\nDownloaded {success_count}/{total_count} base models successfully")
    return success_count == total_count


def download_custom_models():
    """Download custom trained models if available."""
    print("\nDownloading custom models...")
    
    success_count = 0
    total_count = len(CUSTOM_MODELS)
    
    for model_name, model_info in CUSTOM_MODELS.items():
        destination = Path(model_info["destination"])
        
        if destination.exists():
            print(f"Custom model {model_name} already exists, skipping...")
            success_count += 1
            continue
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Check if URL is accessible
            response = requests.head(model_info["url"], timeout=10)
            if response.status_code != 200:
                print(f"Custom model {model_name} not available (HTTP {response.status_code})")
                continue
        except:
            print(f"Custom model {model_name} not accessible")
            continue
        
        if download_file(
            model_info["url"], 
            destination, 
            model_info.get("md5")
        ):
            print(f"✓ Successfully downloaded custom model {model_name}")
            success_count += 1
        else:
            print(f"✗ Failed to download custom model {model_name}")
    
    print(f"Downloaded {success_count}/{total_count} custom models successfully")


def create_model_info_file():
    """Create a model info file with metadata."""
    model_info = {
        "models": {
            "yolo": {
                "path": "models/yolo/best.pt",
                "type": "detection",
                "framework": "ultralytics",
                "input_size": [640, 640],
                "classes": ["embryo_stage1", "embryo_stage2", "embryo_stage3", 
                           "embryo_stage4", "embryo_stage5", "blastocyst"]
            },
            "vit": {
                "path": "models/vit/model.onnx",
                "type": "classification", 
                "framework": "transformers",
                "input_size": [224, 224],
                "classes": ["Grade_A_Excellent", "Grade_B_Good", "Grade_C_Fair",
                           "Grade_D_Poor", "Grade_E_Very_Poor", "Grade_F_Non_Viable"]
            }
        },
        "version": "1.0.0",
        "updated": "2024-01-15"
    }
    
    import json
    with open("models/model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("Created model info file: models/model_info.json")


def verify_installation():
    """Verify that all required models are present."""
    print("\nVerifying installation...")
    
    required_files = [
        "models/yolo/yolov8n.pt",  # Base model
        "models/model_info.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Warning: Some files are missing:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nThe system will use fallback models during runtime.")
    else:
        print("✓ All base models are present")
    
    print("\n🚀 Model download complete! You can now start the EmbryoVision API.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download EmbryoVision models")
    parser.add_argument("--custom-only", action="store_true", 
                       help="Only download custom trained models")
    parser.add_argument("--base-only", action="store_true",
                       help="Only download base models")
    
    args = parser.parse_args()
    
    print("🔬 EmbryoVision Model Downloader")
    print("=" * 40)
    
    # Create directories
    create_directories()
    
    # Download models based on arguments
    if args.custom_only:
        download_custom_models()
    elif args.base_only:
        download_base_models()
    else:
        download_base_models()
        download_custom_models()
    
    # Create model info file
    create_model_info_file()
    
    # Verify installation
    verify_installation()


if __name__ == "__main__":
    main() 