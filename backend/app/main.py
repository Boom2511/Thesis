import time
import io
import sys
import logging
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List

# Create logs directory if not exists
Path('logs').mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/api_{datetime.now():%Y%m%d}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from services.detection import EnsembleDetectionService
from middleware.rate_limit import rate_limit_middleware

# Import video API
try:
    from api import video as video_api
    VIDEO_API_AVAILABLE = True
except ImportError as e:
    print(f"[!] Video API not available: {e}")
    VIDEO_API_AVAILABLE = False
    video_api = None

service: EnsembleDetectionService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    logger.info("Starting Ensemble Deepfake Detection Service...")
    try:
        service = EnsembleDetectionService(config_path="app/config.json")

        # Set service for video API if available
        if VIDEO_API_AVAILABLE and video_api:
            video_api.set_service(service)

        logger.info("Service ready!")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise
    yield
    logger.info("Shutting down service.")

app = FastAPI(
    title="Ensemble Deepfake Detection API",
    version="3.0.0",
    description="Advanced deepfake detection with >90% accuracy. Supports images, videos, and webcam. Includes Grad-CAM explanations.",
    lifespan=lifespan
)

# Include video router if available
if VIDEO_API_AVAILABLE and video_api:
    app.include_router(video_api.router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Rate limiting
app.middleware("http")(rate_limit_middleware)

class DetectionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float
    fake_probability: float
    real_probability: float
    processing_time: float
    face_detection_confidence: Optional[float]
    gradcam: Optional[str]
    heatmap_analysis: Optional[Dict] = None
    model_predictions: Dict
    models_used: List[str]
    total_models: int
    device: str
    error: Optional[str] = None

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Ensemble Deepfake Detection API",
        "version": "3.0.0",
        "status": "running",
        "models": service.model_manager.models.keys() if service else []
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(service.model_manager.models.keys()) if service else [],
        "total_models": len(service.model_manager.models) if service else 0,
        "device": str(service.model_manager.device) if service else "unknown"
    }

@app.post("/api/detect/image", response_model=DetectionResponse, tags=["Detection"])
async def detect_image(
    request: Request,
    file: UploadFile = File(...),
    generate_heatmap: bool = Query(True, description="Generate Grad-CAM heatmap")
):
    """
    Ensemble deepfake detection using 3 models

    - **file**: Image file (JPG, PNG, etc.)
    - **generate_heatmap**: Generate Grad-CAM visualization

    Returns ensemble prediction with individual model results
    """
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"[{client_ip}] New request: {file.filename} ({file.content_type})")

    if not file.content_type.startswith("image/"):
        logger.warning(f"[{client_ip}] Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        start_time = time.time()

        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)

        if len(contents) > 10 * 1024 * 1024:
            logger.warning(f"[{client_ip}] File too large: {file_size_mb:.2f}MB")
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        logger.info(f"[{client_ip}] Processing {file.filename} ({file_size_mb:.2f}MB)")

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        if image.size[0] < 100 or image.size[1] < 100:
            logger.warning(f"[{client_ip}] Image too small: {image.size}")
            raise HTTPException(status_code=400, detail="Image too small (min 100x100)")

        # Process with ensemble
        result = service.process(image, generate_heatmap=generate_heatmap)

        result["filename"] = file.filename
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time

        logger.info(
            f"[{client_ip}] Prediction: {result['prediction']} "
            f"(confidence: {result['confidence']:.2%}, time: {processing_time:.2f}s)"
        )

        return result

    except ValueError as e:
        logger.error(f"[{client_ip}] ValueError: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"[{client_ip}] Error processing {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/models", tags=["Info"])
async def get_models_info():
    """Get information about loaded models"""
    if not service:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "models": list(service.model_manager.models.keys()),
        "config": service.model_manager.config["models"],
        "ensemble_method": service.model_manager.config["ensemble"]["method"]
    }