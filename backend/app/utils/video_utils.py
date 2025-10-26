"""
Video Processing Utilities
Memory-efficient video frame extraction and processing
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class VideoFrameLoader:
    """
    Memory-efficient video frame loader
    Loads frames on-demand without keeping entire video in memory
    """

    def __init__(self, video_path: str):
        """
        Initialize frame loader

        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self._validate_video()

    def _validate_video(self):
        """Validate video can be opened"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        cap.release()

    def get_frame(self, frame_number: int) -> Image.Image:
        """
        Load a specific frame without keeping video open

        Args:
            frame_number: Frame index to load (0-based)

        Returns:
            PIL Image in RGB format

        Raises:
            ValueError: If frame cannot be loaded
        """
        cap = cv2.VideoCapture(self.video_path)

        try:
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if not ret or frame is None:
                raise ValueError(f"Could not load frame {frame_number}")

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)

        finally:
            cap.release()

    def get_frames(self, frame_numbers: List[int]) -> List[Image.Image]:
        """
        Load multiple specific frames efficiently

        Args:
            frame_numbers: List of frame indices to load

        Returns:
            List of PIL Images in RGB format
        """
        frames = []

        for frame_num in sorted(frame_numbers):
            try:
                frame = self.get_frame(frame_num)
                frames.append(frame)
            except ValueError as e:
                logger.warning(f"Skipping frame {frame_num}: {e}")

        return frames

    def get_video_info(self) -> Dict[str, any]:
        """
        Get video metadata

        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(self.video_path)

        try:
            info = {
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': 0.0
            }

            if info['fps'] > 0:
                info['duration'] = info['total_frames'] / info['fps']

            return info

        finally:
            cap.release()


class AdaptiveFrameSampler:
    """
    Intelligently sample frames from video based on content and duration
    """

    @staticmethod
    def calculate_optimal_skip(
        duration: float,
        fps: float,
        max_frames: int = 100,
        min_frames: int = 10
    ) -> Tuple[int, int]:
        """
        Calculate optimal frame skip and max frames based on video duration

        Args:
            duration: Video duration in seconds
            fps: Video frames per second
            max_frames: Maximum frames to process
            min_frames: Minimum frames to process

        Returns:
            Tuple of (frame_skip, adjusted_max_frames)
        """
        total_frames = int(duration * fps)

        if total_frames <= min_frames:
            # Very short video - process all frames
            return 1, total_frames

        # Calculate frame skip to stay under max_frames
        frame_skip = max(1, total_frames // max_frames)

        # Ensure we get at least min_frames
        if total_frames // frame_skip < min_frames:
            frame_skip = max(1, total_frames // min_frames)

        adjusted_max = min(max_frames, total_frames // frame_skip)

        return frame_skip, adjusted_max

    @staticmethod
    def get_key_frame_indices(
        frame_results: List[Dict],
        top_n: int = 3,
        criteria: str = 'fake_probability'
    ) -> List[int]:
        """
        Get indices of most important frames based on criteria

        Args:
            frame_results: List of frame analysis results
            top_n: Number of top frames to return
            criteria: Sorting criteria ('fake_probability' or 'confidence')

        Returns:
            List of frame indices
        """
        if not frame_results:
            return []

        # Sort by criteria (descending)
        sorted_frames = sorted(
            frame_results,
            key=lambda x: x.get(criteria, 0),
            reverse=True
        )

        # Get top N frame numbers
        top_frames = sorted_frames[:top_n]
        return [f['frame_number'] for f in top_frames]


class VideoProcessingOptimizer:
    """
    Optimization strategies for video processing
    """

    @staticmethod
    def estimate_memory_usage(
        width: int,
        height: int,
        frames_to_process: int,
        store_frames: bool = False
    ) -> Dict[str, float]:
        """
        Estimate memory usage for video processing

        Args:
            width: Frame width
            height: Frame height
            frames_to_process: Number of frames to process
            store_frames: Whether frames are stored in memory

        Returns:
            Dictionary with memory estimates in MB
        """
        # Each RGB frame: width * height * 3 bytes
        frame_size_bytes = width * height * 3

        # Processing overhead (tensors, models, etc.)
        processing_overhead_mb = 500  # ~500MB for models

        # Frame storage if needed
        frame_storage_mb = 0
        if store_frames:
            frame_storage_mb = (frame_size_bytes * frames_to_process) / (1024 ** 2)

        # Temporary processing buffer (3 frames at once)
        temp_buffer_mb = (frame_size_bytes * 3) / (1024 ** 2)

        total_mb = processing_overhead_mb + frame_storage_mb + temp_buffer_mb

        return {
            'model_overhead_mb': processing_overhead_mb,
            'frame_storage_mb': frame_storage_mb,
            'temp_buffer_mb': temp_buffer_mb,
            'total_estimated_mb': total_mb,
            'recommendation': 'OK' if total_mb < 2048 else 'REDUCE_FRAMES'
        }

    @staticmethod
    def should_reduce_quality(video_info: Dict) -> bool:
        """
        Determine if video processing should use reduced quality for performance

        Args:
            video_info: Video metadata dictionary

        Returns:
            True if should reduce quality
        """
        # High resolution videos (> 1080p)
        is_high_res = video_info.get('width', 0) > 1920

        # Long videos (> 5 minutes)
        is_long = video_info.get('duration', 0) > 300

        # High frame rate (> 60fps)
        is_high_fps = video_info.get('fps', 0) > 60

        return is_high_res or is_long or is_high_fps
