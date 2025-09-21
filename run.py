#!/usr/bin/env python3
"""
EmbryoVision Application Runner

This script provides convenient ways to run the EmbryoVision application
in different modes (development, production, etc.)
"""

import argparse
import sys
import subprocess
import os
from pathlib import Path


def run_dev():
    """Run in development mode with hot reload."""
    print("🚀 Starting EmbryoVision in development mode...")
    cmd = [
        "uvicorn", 
        "app.main:app", 
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
        "--log-level", "debug"
    ]
    subprocess.run(cmd)


def run_prod():
    """Run in production mode."""
    print("🏭 Starting EmbryoVision in production mode...")
    cmd = [
        "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0", 
        "--port", "8000",
        "--workers", "4",
        "--log-level", "info"
    ]
    subprocess.run(cmd)


def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    cmd = ["pytest", "-v", "--cov=app", "--cov-report=html"]
    subprocess.run(cmd)


def download_models():
    """Download pre-trained models."""
    print("📥 Downloading models...")
    cmd = ["python", "scripts/download_models.py"]
    subprocess.run(cmd)


def optimize_models():
    """Optimize models for production."""
    print("⚡ Optimizing models...")
    cmd = ["python", "-m", "app.models.optimization", "--all"]
    subprocess.run(cmd)


def docker_build():
    """Build Docker containers."""
    print("🐳 Building Docker containers...")
    cmd = ["docker-compose", "build"]
    subprocess.run(cmd)


def docker_up():
    """Start Docker stack."""
    print("🐳 Starting Docker stack...")
    cmd = ["docker-compose", "up", "-d"]
    subprocess.run(cmd)


def docker_down():
    """Stop Docker stack."""
    print("🐳 Stopping Docker stack...")
    cmd = ["docker-compose", "down"]
    subprocess.run(cmd)


def check_deps():
    """Check if all dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA: {torch.version.cuda}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available")
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    try:
        import fastapi
        print(f"✓ FastAPI: {fastapi.__version__}")
    except ImportError:
        print("❌ FastAPI not installed")
        return False
    
    try:
        import ultralytics
        print(f"✓ Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics not installed")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
        return False
    
    # Check if models exist
    models_dir = Path("models")
    if models_dir.exists():
        yolo_models = list(models_dir.glob("yolo/*.pt"))
        vit_models = list(models_dir.glob("vit/*.onnx")) + list(models_dir.glob("vit/*.pt"))
        
        if yolo_models:
            print(f"✓ YOLO models found: {len(yolo_models)}")
        else:
            print("⚠️  No YOLO models found")
        
        if vit_models:
            print(f"✓ ViT models found: {len(vit_models)}")
        else:
            print("⚠️  No ViT models found")
    else:
        print("⚠️  Models directory not found")
    
    print("\n✅ Dependency check complete!")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="EmbryoVision Application Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py dev              # Start development server
  python run.py prod             # Start production server  
  python run.py test             # Run tests
  python run.py download-models  # Download pre-trained models
  python run.py check            # Check dependencies
  python run.py docker-up        # Start with Docker
        """
    )
    
    parser.add_argument(
        "command",
        choices=[
            "dev", "prod", "test", "download-models", 
            "optimize-models", "docker-build", "docker-up", 
            "docker-down", "check"
        ],
        help="Command to run"
    )
    
    args = parser.parse_args()
    
    # Set up environment
    os.environ["PYTHONPATH"] = str(Path.cwd())
    
    # Print banner
    print("🔬 EmbryoVision - Real-Time Embryo Quality Scoring")
    print("=" * 50)
    
    # Execute command
    if args.command == "dev":
        run_dev()
    elif args.command == "prod":
        run_prod()
    elif args.command == "test":
        run_tests()
    elif args.command == "download-models":
        download_models()
    elif args.command == "optimize-models":
        optimize_models()
    elif args.command == "docker-build":
        docker_build()
    elif args.command == "docker-up":
        docker_up()
    elif args.command == "docker-down":
        docker_down()
    elif args.command == "check":
        check_deps()


if __name__ == "__main__":
    main() 