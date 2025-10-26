# -*- coding: utf-8 -*-
"""
Video Processing API
Goal 1.2.3: Support video upload and webcam detection
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request, Header
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from typing import List, Dict, Optional
import json
from services.detection import EnsembleDetectionService
import time
import logging

# Import rate limiter
try:
    from middleware.rate_limit import limiter, RateLimits
    RATE_LIMIT_AVAILABLE = True
except ImportError:
    RATE_LIMIT_AVAILABLE = False
    limiter = None
    RateLimits = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/video", tags=["Video Detection"])

# Global service (will be injected)
service: EnsembleDetectionService = None


def set_service(detection_service: EnsembleDetectionService):
    """Set the detection service"""
    global service
    service = detection_service


@router.post("/detect")
async def detect_video(
    request: Request,
    file: UploadFile = File(...),
    frame_skip: int = Query(5, ge=1, le=30, description="Process every Nth frame"),
    max_frames: int = Query(100, ge=10, le=500, description="Maximum frames to process")
):
    """
    Detect deepfakes in video

    Args:
        file: Video file (MP4, AVI, MOV, etc.)
        frame_skip: Process every Nth frame (1=all frames, max 30)
        max_frames: Maximum number of frames to process (10-500)

    Returns:
        Video analysis results with key frame heatmaps
    """
    # Note: Rate limiting available when slowapi is installed
    # See SECURITY_AND_PERFORMANCE_GUIDE.md for setup

    # Validate file type
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be video.")

    # Validate file size (max 100MB)
    MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB
    content = await file.read()

    if len(content) > MAX_VIDEO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Video file too large. Maximum size: {MAX_VIDEO_SIZE / (1024*1024):.0f}MB"
        )

    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Process video
        results = await process_video_file(
            tmp_path,
            frame_skip=frame_skip,
            max_frames=max_frames
        )

        return results

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


async def process_video_file(video_path: str,
                              frame_skip: int = 5,
                              max_frames: int = 100,
                              generate_key_frame_heatmaps: bool = True) -> Dict:
    """
    Process video file and detect deepfakes

    Args:
        video_path: Path to video file
        frame_skip: Process every Nth frame
        max_frames: Maximum frames to process
        generate_key_frame_heatmaps: Generate Grad-CAM for key frames

    Returns:
        Analysis results with key frame heatmaps
    """
    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Failed to open video file")

    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    frame_results = []
    frame_images = []  # Store frames for key frame selection
    frame_count = 0
    processed_count = 0

    start_time = time.time()

    try:
        while cap.isOpened() and processed_count < max_frames:
            ret, frame = cap.read()

            if not ret:
                break

            # Skip frames
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Process frame
            try:
                result = service.process(pil_image, generate_heatmap=False)

                frame_results.append({
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps if fps > 0 else 0,
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'fake_probability': result['fake_probability'],
                    'model_predictions': result['model_predictions']
                })

                # Store frame for potential Grad-CAM generation
                frame_images.append((frame_count, pil_image, result['fake_probability']))

                processed_count += 1

            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")

            frame_count += 1

    finally:
        cap.release()

    processing_time = time.time() - start_time

    # Analyze results
    fake_frames = sum(1 for r in frame_results if r['prediction'] == 'FAKE')
    real_frames = len(frame_results) - fake_frames

    fake_ratio = fake_frames / len(frame_results) if frame_results else 0
    avg_confidence = np.mean([r['confidence'] for r in frame_results]) if frame_results else 0

    # Overall verdict
    overall_prediction = "FAKE" if fake_ratio > 0.5 else "REAL"
    overall_confidence = fake_ratio if overall_prediction == "FAKE" else (1 - fake_ratio)

    # Generate Grad-CAM for key frames (top 3 most suspicious)
    key_frame_heatmaps = []
    if generate_key_frame_heatmaps and frame_images:
        # Sort by fake probability (descending)
        sorted_frames = sorted(frame_images, key=lambda x: x[2], reverse=True)
        top_frames = sorted_frames[:min(3, len(sorted_frames))]

        for frame_num, frame_img, fake_prob in top_frames:
            try:
                # Generate Grad-CAM for this frame
                result = service.process(frame_img, generate_heatmap=True)
                key_frame_heatmaps.append({
                    'frame_number': frame_num,
                    'timestamp': frame_num / fps if fps > 0 else 0,
                    'fake_probability': fake_prob,
                    'gradcam': result.get('gradcam')
                })
            except Exception as e:
                print(f"Failed to generate Grad-CAM for frame {frame_num}: {e}")

    result_data = {
        'video_info': {
            'filename': os.path.basename(video_path),
            'total_frames': total_frames,
            'fps': fps,
            'duration_seconds': duration,
            'resolution': f"{width}x{height}"
        },
        'processing_info': {
            'frames_processed': processed_count,
            'frame_skip': frame_skip,
            'processing_time': processing_time,
            'processing_fps': processed_count / processing_time if processing_time > 0 else 0
        },
        'overall_result': {
            'prediction': overall_prediction,
            'confidence': overall_confidence,
            'fake_frame_ratio': fake_ratio,
            'total_fake_frames': fake_frames,
            'total_real_frames': real_frames
        },
        'frame_results': frame_results,
        'key_frame_heatmaps': key_frame_heatmaps,
        'summary': {
            'avg_confidence': avg_confidence,
            'max_fake_confidence': max([r['fake_probability'] for r in frame_results], default=0),
            'min_fake_confidence': min([r['fake_probability'] for r in frame_results], default=0)
        }
    }

    # Log to MLflow
    if service.mlflow:
        try:
            service.mlflow.log_prediction("video", result_data, {
                'filename': os.path.basename(video_path),
                'duration': duration,
                'frames_processed': processed_count
            })
        except Exception as e:
            print(f"⚠️  MLflow logging failed: {e}")

    return result_data


@router.get("/stream_info")
async def get_stream_info():
    """Get information about webcam streaming capability"""
    return {
        'webcam_support': True,
        'recommended_settings': {
            'frame_rate': 15,
            'resolution': '640x480',
            'format': 'JPEG'
        },
        'note': 'Use /api/detect/image endpoint for individual frame analysis'
    }


# WebSocket support for real-time webcam detection
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

# Import authentication
try:
    from middleware.auth import verify_websocket_token
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False


@router.websocket("/ws/webcam")
async def websocket_webcam(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="Session token")
):
    """
    WebSocket endpoint for real-time webcam detection

    **Security**: Session token recommended for production

    Usage:
        ws://localhost:8000/api/video/ws/webcam?token=YOUR_TOKEN

    Client sends: Base64 encoded JPEG frames
    Server responds: Detection results in JSON
    """
    await websocket.accept()

    # Verify authentication (optional in dev mode)
    if AUTH_AVAILABLE and token:
        is_valid = await verify_websocket_token(websocket, token)
        if not is_valid:
            logger.warning("WebSocket rejected: Invalid token")
            return

    logger.info("WebSocket connected")

    # Max frame size: 5MB
    MAX_FRAME_SIZE = 5 * 1024 * 1024
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Receive frame
            data = await websocket.receive_text()

            # Validate size
            if len(data) > MAX_FRAME_SIZE:
                await websocket.send_json({
                    'error': 'Frame too large (max 5MB)',
                    'timestamp': time.time()
                })
                continue

            # Process frame
            try:
                import base64
                from io import BytesIO

                # Remove data URI prefix
                if data.startswith('data:image'):
                    data = data.split(',')[1]

                # Decode
                image_data = base64.b64decode(data)
                image = Image.open(BytesIO(image_data)).convert('RGB')

                # Detect
                result = service.process(image, generate_heatmap=False)

                frame_count += 1

                # Send result
                await websocket.send_json({
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'fake_probability': result['fake_probability'],
                    'real_probability': result['real_probability'],
                    'frame_count': frame_count,
                    'timestamp': time.time()
                })

            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                await websocket.send_json({
                    'error': str(e),
                    'timestamp': time.time()
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected ({frame_count} frames)")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011)


# Batch video processing
@router.post("/detect_batch")
async def detect_video_batch(
    files: List[UploadFile] = File(...),
    frame_skip: int = Query(10, description="Process every Nth frame"),
    max_frames: int = Query(50, description="Maximum frames per video")
):
    """
    Process multiple videos in batch

    Args:
        files: List of video files
        frame_skip: Process every Nth frame
        max_frames: Maximum frames per video

    Returns:
        Batch processing results
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 videos per batch"
        )

    results = []

    for file in files:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Process video
            video_result = await process_video_file(
                tmp_path,
                frame_skip=frame_skip,
                max_frames=max_frames
            )

            video_result['filename'] = file.filename

            results.append(video_result)

        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return {
        'total_videos': len(files),
        'results': results,
        'summary': {
            'fake_videos': sum(1 for r in results if r.get('overall_result', {}).get('prediction') == 'FAKE'),
            'real_videos': sum(1 for r in results if r.get('overall_result', {}).get('prediction') == 'REAL'),
            'errors': sum(1 for r in results if 'error' in r)
        }
    }
