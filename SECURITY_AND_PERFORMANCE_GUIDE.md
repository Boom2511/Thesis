# 🛡️ Security & Performance Guide

## Table of Contents
1. [Security Improvements](#security-improvements)
2. [Performance Optimization](#performance-optimization)
3. [MLflow Usage Guide](#mlflow-usage-guide)
4. [Environment Configuration](#environment-configuration)
5. [Best Practices](#best-practices)

---

## 🔒 Security Improvements

### 1. Rate Limiting

**Implementation**: `backend/app/middleware/rate_limit.py`

#### Configuration:
```python
from middleware.rate_limit import limiter, RateLimits

# Apply to endpoints
@router.post("/detect")
@limiter.limit(RateLimits.VIDEO_DETECT)  # 2 requests/minute
async def detect_video(request: Request, ...):
    ...
```

#### Default Limits:
- **Image Detection**: 10 requests/minute
- **Batch Processing**: 3 requests/minute
- **Video Detection**: 2 requests/minute
- **Video Batch**: 1 request/5 minutes
- **WebSocket**: 5 connections/hour

#### Production Setup:
```python
# Use Redis for distributed rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379"
)
```

---

### 2. WebSocket Authentication

**Implementation**: `backend/app/middleware/auth.py`

#### How to Use:

**Step 1**: Generate session token (add this endpoint):
```python
@router.get("/api/video/session")
async def create_session():
    token = SessionManager.generate_token()
    return {"token": token, "expires_in": 3600}
```

**Step 2**: Connect with token:
```javascript
const ws = new WebSocket(
    `ws://localhost:8000/api/video/ws/webcam?token=${token}`
);
```

#### Security Features:
- ✅ Token expiration (1 hour)
- ✅ Per-session rate limiting (100 requests/hour)
- ✅ Automatic cleanup of expired sessions
- ✅ Frame size validation (max 5MB)

---

### 3. Input Validation

#### Video Upload Protection:
```python
# File type validation
if not file.content_type.startswith("video/"):
    raise HTTPException(400, "Invalid file type")

# File size limit (100MB)
if len(content) > 100 * 1024 * 1024:
    raise HTTPException(413, "File too large")

# Parameter validation
frame_skip: int = Query(5, ge=1, le=30)  # 1-30
max_frames: int = Query(100, ge=10, le=500)  # 10-500
```

---

## ⚡ Performance Optimization

### 1. Memory Management - Video Processing

**Problem**: Original code stored all frames in memory
```python
# ❌ BAD: Consumes GB of RAM
frame_images.append((frame_count, pil_image, ...))
```

**Solution**: Store only metadata, load frames on-demand

```python
# ✅ GOOD: Memory efficient
frame_metadata = []
frame_metadata.append({
    'frame_number': frame_count,
    'fake_probability': result['fake_probability']
})

# Load specific frames later for Grad-CAM
def load_frame_at_position(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame
```

### Implementation:
Create `backend/app/utils/video_utils.py`:

```python
import cv2
from PIL import Image
import numpy as np

class VideoFrameLoader:
    """Memory-efficient video frame loader"""

    def __init__(self, video_path: str):
        self.video_path = video_path

    def get_frame(self, frame_number: int) -> Image.Image:
        """Load a specific frame without keeping video open"""
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not load frame {frame_number}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
```

---

### 2. Async MLflow Logging

**Problem**: MLflow logging blocks requests

**Solution**: Use background tasks

```python
from fastapi import BackgroundTasks

async def log_to_mlflow_async(result, metadata):
    """Background MLflow logging"""
    if service.mlflow:
        try:
            service.mlflow.log_prediction("video", result, metadata)
        except Exception as e:
            logger.error(f"MLflow logging failed: {e}")

@router.post("/detect")
async def detect_video(
    request: Request,
    background_tasks: BackgroundTasks,
    ...
):
    # ... process video ...

    # Log asynchronously
    background_tasks.add_task(
        log_to_mlflow_async,
        result_data,
        metadata
    )

    return result_data
```

---

### 3. Video Processing Optimization

#### Frame Skip Strategy:
```python
# Low-quality preview (fast)
frame_skip = 30  # Every 30th frame (~1 per second at 30fps)

# Balanced (default)
frame_skip = 5   # Every 5th frame (~6 per second at 30fps)

# High-quality (slow)
frame_skip = 1   # Every frame
```

#### Processing Limits:
```python
# Short videos: Full analysis
if duration < 10:  # Less than 10 seconds
    max_frames = 300  # Process all frames

# Long videos: Sample frames
elif duration < 60:  # Less than 1 minute
    max_frames = 100  # ~10 seconds worth

else:  # Very long videos
    max_frames = 50   # Quick analysis
```

---

## 📊 MLflow Usage Guide

### Installation & Setup

```bash
cd backend
pip install mlflow>=2.10.0
```

### Starting MLflow UI

```bash
# Start MLflow tracking server
cd backend
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

**Access UI**: http://localhost:5000

---

### How MLflow is Integrated

#### 1. Automatic Logging (Already Implemented)

Every prediction is automatically logged:

```python
# Image detection
service.process(image, metadata={'filename': 'test.jpg'})
# ✅ Automatically logs to MLflow

# Video detection
# ✅ Automatically logs video results + metadata
```

#### 2. What Gets Logged

**For Images**:
- Prediction (FAKE/REAL)
- Confidence score
- Fake/real probabilities
- Processing time
- Each model's prediction
- Device used (CPU/GPU)
- Custom metadata (filename, source, etc.)

**For Videos**:
- Overall verdict
- Frame statistics
- Processing performance (FPS)
- Fake frame ratio
- Average confidence
- Video metadata (duration, resolution)

#### 3. Viewing Results

**MLflow UI Sections**:

1. **Experiments**
   - View all runs
   - Compare predictions
   - Filter by prediction type

2. **Metrics**
   - Confidence scores
   - Processing times
   - Model performance

3. **Parameters**
   - Prediction type (image/video/webcam)
   - Models used
   - Configuration

---

### MLflow Queries & Analysis

#### Get Experiment Statistics:
```python
from services.mlflow_service import MLflowService

mlflow_service = MLflowService()
stats = mlflow_service.get_experiment_stats()

print(f"Total runs: {stats['total_runs']}")
print(f"Average confidence: {stats['avg_confidence']}")
print(f"Fake detections: {stats['total_fake']}")
print(f"Real detections: {stats['total_real']}")
```

#### Query Predictions:
```python
import mlflow

# Get all runs
runs = mlflow.search_runs(
    experiment_names=["deepfake_detection"]
)

# Filter high-confidence fakes
fake_runs = runs[
    (runs['params.prediction'] == 'FAKE') &
    (runs['metrics.confidence'] > 0.9)
]

print(f"High-confidence fakes: {len(fake_runs)}")
```

#### Compare Models:
```python
# Get average confidence per model
xception_conf = runs['metrics.xception_confidence'].mean()
effort_conf = runs['metrics.effort_fake_prob'].mean()

print(f"Xception avg: {xception_conf:.2%}")
print(f"Effort avg: {effort_conf:.2%}")
```

---

### MLflow Best Practices

#### 1. Organization
```python
# Use descriptive run names
mlflow.start_run(run_name=f"video_{filename}_{timestamp}")

# Tag important runs
mlflow.set_tag("dataset", "production")
mlflow.set_tag("model_version", "v3.0")
```

#### 2. Batch Logging
```python
# Log batch statistics
mlflow_service.log_batch_results(
    results=batch_results,
    batch_type="image"
)
```

#### 3. Model Tracking
```python
# Log model configuration
mlflow_service.log_model_info(config)
```

---

## 🔧 Environment Configuration

Create `.env` file:

```bash
# Backend Configuration
API_URL=http://localhost:8000
FRONTEND_URL=http://localhost:3000

# Security
SECRET_KEY=your-secret-key-here-change-in-production
API_KEY_REQUIRED=false  # Set to true in production

# MLflow
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=deepfake_detection

# Rate Limiting
REDIS_URL=redis://localhost:6379  # For production
RATE_LIMIT_STORAGE=memory  # or 'redis'

# Processing Limits
MAX_VIDEO_SIZE_MB=100
MAX_IMAGE_SIZE_MB=10
MAX_BATCH_SIZE=10

# Model Configuration
DEVICE=cpu  # or 'cuda'
ENABLE_GRADCAM=true
ENABLE_MLFLOW=true
```

### Load in Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()

API_URL = os.getenv('API_URL', 'http://localhost:8000')
MAX_VIDEO_SIZE = int(os.getenv('MAX_VIDEO_SIZE_MB', 100)) * 1024 * 1024
```

### Load in Frontend:
```typescript
// frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

```typescript
// Usage
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
```

---

## ✅ Best Practices Checklist

### Security
- [ ] Enable rate limiting in production
- [ ] Use Redis for distributed rate limiting
- [ ] Implement API key authentication
- [ ] Enable WebSocket token validation
- [ ] Set up HTTPS in production
- [ ] Configure CORS properly
- [ ] Add request logging
- [ ] Implement input sanitization

### Performance
- [ ] Use async MLflow logging
- [ ] Implement video frame pagination
- [ ] Add caching for repeated requests
- [ ] Monitor memory usage
- [ ] Use GPU if available
- [ ] Implement request queuing for heavy loads

### Monitoring
- [ ] Set up MLflow tracking server
- [ ] Monitor processing times
- [ ] Track error rates
- [ ] Log system metrics (CPU, memory)
- [ ] Set up alerts for anomalies

### Code Quality
- [ ] Add comprehensive tests
- [ ] Use type hints
- [ ] Write documentation
- [ ] Follow PEP 8 style guide
- [ ] Add error handling
- [ ] Use logging instead of print

---

## 🚀 Production Deployment

### 1. Security Hardening
```python
# main.py
app = FastAPI(
    title="Deepfake Detection API",
    docs_url=None if PRODUCTION else "/docs",  # Disable docs in prod
    redoc_url=None if PRODUCTION else "/redoc"
)

# Enable HTTPS only
app.add_middleware(
    HTTPSRedirectMiddleware
)
```

### 2. Performance Tuning
```bash
# Use gunicorn with multiple workers
gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300 \
    --max-requests 1000 \
    --max-requests-jitter 100
```

### 3. Monitoring Setup
```bash
# Start MLflow server with database backend
mlflow server \
    --backend-store-uri postgresql://user:pass@localhost/mlflow \
    --default-artifact-root s3://my-mlflow-bucket/ \
    --host 0.0.0.0 \
    --port 5000
```

---

## 📝 Summary

### Security Enhancements
✅ Rate limiting with configurable limits
✅ WebSocket authentication
✅ Input validation (file size, type, parameters)
✅ Frame size limits for WebSocket
✅ Session management with expiration

### Performance Optimizations
✅ Memory-efficient video processing
✅ Async MLflow logging
✅ Frame skipping strategies
✅ Processing limits
✅ Background tasks

### MLflow Integration
✅ Automatic experiment tracking
✅ Comprehensive metrics logging
✅ Batch analysis support
✅ Query and comparison tools
✅ Web UI for visualization

### Configuration
✅ Environment-based settings
✅ Production-ready defaults
✅ Flexible deployment options

---

**Need Help?** Check the code comments or open an issue!
