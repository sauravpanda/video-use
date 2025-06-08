"""Frame extraction from video files."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Iterator
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from ..schema.models import VideoMetadata, Frame, VideoAnalysisConfig

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extracts frames from video files with intelligent frame selection."""
    
    def __init__(self, config: VideoAnalysisConfig):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
    
    async def extract_video_metadata(self, video_path: Path) -> VideoMetadata:
        """Extract basic metadata from video file."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Get file info
            file_size = video_path.stat().st_size
            file_format = video_path.suffix.lower()
            
            cap.release()
            
            return VideoMetadata(
                file_path=video_path,
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                total_frames=frame_count,
                format=file_format,
                size_bytes=file_size
            )
            
        except Exception as e:
            logger.error(f"Error extracting video metadata: {e}")
            raise
    
    async def extract_frames(self, video_path: Path) -> List[Frame]:
        """Extract frames from video based on configuration."""
        try:
            metadata = await self.extract_video_metadata(video_path)
            
            # Calculate frame extraction parameters
            frame_interval = max(1, int(metadata.fps / self.config.frame_extraction_fps))
            max_frames = min(self.config.max_frames, metadata.total_frames // frame_interval)
            
            logger.info(f"Extracting {max_frames} frames from {video_path}")
            logger.info(f"Frame interval: {frame_interval}, FPS: {metadata.fps}")
            
            # Extract frames in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            frames = await loop.run_in_executor(
                self.executor, 
                self._extract_frames_sync, 
                video_path, 
                frame_interval, 
                max_frames
            )
            
            # Filter frames based on visual differences
            if self.config.min_frame_difference > 0:
                original_count = len(frames)
                frames = await self._filter_frames_by_difference(frames)
                logger.info(f"Frame filtering: {original_count} -> {len(frames)} frames (threshold: {self.config.min_frame_difference})")
            
            logger.info(f"Successfully extracted {len(frames)} frames")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            raise
    
    def _extract_frames_sync(self, video_path: Path, frame_interval: int, max_frames: int) -> List[Frame]:
        """Synchronous frame extraction (runs in thread pool)."""
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        
        try:
            frame_count = 0
            extracted_count = 0
            
            while cap.isOpened() and extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at specified intervals
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / cap.get(cv2.CAP_PROP_FPS)
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    frames.append(Frame(
                        frame_number=frame_count,
                        timestamp=timestamp,
                        image=frame_rgb,
                        is_keyframe=False,  # Will be determined later
                        visual_diff_score=0.0
                    ))
                    
                    extracted_count += 1
                
                frame_count += 1
            
            logger.debug(f"Frame extraction sync: processed {frame_count} total frames, extracted {extracted_count} frames")
            return frames
            
        finally:
            cap.release()
    
    async def _filter_frames_by_difference(self, frames: List[Frame]) -> List[Frame]:
        """Filter frames based on visual differences."""
        if len(frames) <= 1:
            return frames
        
        filtered_frames = [frames[0]]  # Always keep first frame
        
        loop = asyncio.get_event_loop()
        
        for i in range(1, len(frames)):
            # Calculate visual difference in thread pool
            diff_score = await loop.run_in_executor(
                self.executor,
                self._calculate_frame_difference,
                frames[i-1].image,
                frames[i].image
            )
            
            frames[i].visual_diff_score = diff_score
            
            # Keep frame if difference is significant
            if diff_score >= self.config.min_frame_difference:
                filtered_frames.append(frames[i])
                logger.debug(f"Frame {frames[i].frame_number}: diff={diff_score:.3f} (kept)")
            else:
                logger.debug(f"Frame {frames[i].frame_number}: diff={diff_score:.3f} (filtered)")
        
        # Ensure we have at least a few frames for action detection
        if len(filtered_frames) < 3 and len(frames) >= 3:
            logger.warning(f"Only {len(filtered_frames)} frames after filtering, keeping more frames for action detection")
            # Keep every Nth frame to ensure we have at least 5-10 frames
            step = max(1, len(frames) // 10)
            filtered_frames = frames[::step][:10]
            logger.info(f"Fallback: keeping {len(filtered_frames)} frames with step size {step}")
        
        return filtered_frames
    
    def _calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate visual difference between two frames."""
        try:
            # Convert to grayscale for comparison
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            
            # Calculate structural similarity
            diff = cv2.absdiff(gray1, gray2)
            diff_score = np.mean(diff) / 255.0
            
            return diff_score
            
        except Exception as e:
            logger.warning(f"Error calculating frame difference: {e}")
            return 1.0  # Assume significant difference on error
    
    async def extract_keyframes(self, video_path: Path) -> List[Frame]:
        """Extract keyframes using scene detection."""
        try:
            frames = await self.extract_frames(video_path)
            
            # Mark frames with high visual difference as keyframes
            keyframes = []
            for i, frame in enumerate(frames):
                if i == 0 or frame.visual_diff_score > (self.config.min_frame_difference * 2):
                    frame.is_keyframe = True
                    keyframes.append(frame)
            
            logger.info(f"Identified {len(keyframes)} keyframes out of {len(frames)} total frames")
            return keyframes
            
        except Exception as e:
            logger.error(f"Error extracting keyframes: {e}")
            raise
    
    async def save_frames(self, frames: List[Frame], output_dir: Path) -> List[Frame]:
        """Save frames to disk and update their file paths."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_frames = []
        for frame in frames:
            frame_path = output_dir / f"frame_{frame.frame_number:06d}.jpg"
            
            # Save frame in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._save_frame_sync,
                frame.image,
                frame_path
            )
            
            # Update frame with saved path
            updated_frame = Frame(
                frame_number=frame.frame_number,
                timestamp=frame.timestamp,
                image=frame.image,
                image_path=frame_path,
                is_keyframe=frame.is_keyframe,
                visual_diff_score=frame.visual_diff_score
            )
            saved_frames.append(updated_frame)
        
        logger.info(f"Saved {len(saved_frames)} frames to {output_dir}")
        return saved_frames
    
    def _save_frame_sync(self, frame: np.ndarray, output_path: Path) -> None:
        """Save frame to disk (runs in thread pool)."""
        # Convert RGB back to BGR for saving
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), frame_bgr)
    
    def __del__(self):
        """Clean up thread pool executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False) 