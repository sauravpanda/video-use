"""Video analysis specific services."""

import asyncio
import base64
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from ..models import (
    VideoAnalysisResult, VideoAnalysisConfig, Frame, UIElement, 
    Action, WorkflowDefinition, VideoWorkflowStep, VideoMetadata
)
from ..prompts import GEMINI_VIDEO_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


class VideoAnalysisService:
    """Service for comprehensive video analysis."""
    
    def __init__(self, config: VideoAnalysisConfig):
        self.config = config
        self.frame_service = FrameExtractionService(config)
        logger.info("VideoAnalysisService initialized")
    
    async def analyze_video(self, video_path: Path) -> VideoAnalysisResult:
        """Analyze a video file and extract workflow actions."""
        start_time = time.time()
        
        try:
            logger.info(f"Starting video analysis for: {video_path}")
            
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Extract video metadata
            video_metadata = await self.frame_service.extract_video_metadata(video_path)
            logger.info(f"Video metadata: {video_metadata.duration:.1f}s, {video_metadata.fps:.1f}fps")
            
            # Extract frames
            frames = await self.frame_service.extract_frames(video_path)
            logger.info(f"Extracted {len(frames)} frames")
            
            if not frames:
                return VideoAnalysisResult(
                    video_metadata=video_metadata,
                    frames=[],
                    ui_elements=[],
                    actions=[],
                    success=False,
                    error_message="No frames could be extracted from video"
                )
            
            # For now, create mock UI elements and actions
            # In a full implementation, these would use actual detection services
            ui_elements = []
            actions = self._create_mock_actions(frames)
            
            # Generate workflow definition
            workflow = await self._generate_workflow(actions, frames, video_metadata)
            
            processing_time = time.time() - start_time
            
            return VideoAnalysisResult(
                video_metadata=video_metadata,
                frames=frames,
                ui_elements=ui_elements,
                actions=actions,
                workflow=workflow,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Video analysis failed: {e}")
            
            return VideoAnalysisResult(
                video_metadata=video_metadata if 'video_metadata' in locals() else None,
                frames=frames if 'frames' in locals() else [],
                ui_elements=[],
                actions=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def analyze_video_with_prompt(self, video_path: Path, user_prompt: str) -> VideoAnalysisResult:
        """Analyze video with additional context from user prompt."""
        result = await self.analyze_video(video_path)
        
        if result.workflow:
            result.workflow.description += f" User context: {user_prompt}"
            result.workflow.metadata['user_prompt'] = user_prompt
        
        return result
    
    def _create_mock_actions(self, frames: List[Frame]) -> List[Action]:
        """Create mock actions for demonstration. Replace with actual action inference."""
        from ..models import ActionType
        
        actions = []
        for i, frame in enumerate(frames[:5]):  # Limit to first 5 frames
            action = Action(
                action_type=ActionType.CLICK,
                start_frame=frame.frame_number,
                end_frame=frame.frame_number,
                timestamp=frame.timestamp,
                confidence=0.8,
                description=f"Mock action {i + 1} detected at frame {frame.frame_number}"
            )
            actions.append(action)
        
        return actions
    
    async def _generate_workflow(
        self, 
        actions: List[Action], 
        frames: List[Frame],
        video_metadata: VideoMetadata
    ) -> Optional[WorkflowDefinition]:
        """Generate a workflow definition from the extracted actions."""
        try:
            if not actions:
                logger.warning("No actions found, cannot generate workflow")
                return None
            
            workflow_steps = []
            
            for i, action in enumerate(actions):
                step_id = f"step_{i+1}"
                
                action_frame = None
                for frame in frames:
                    if frame.frame_number == action.start_frame:
                        action_frame = frame
                        break
                
                step = VideoWorkflowStep(
                    step_id=step_id,
                    action=action,
                    description=action.description or f"{action.action_type.value} action",
                    video_timestamp=action.timestamp,
                    confidence_score=action.confidence,
                    frame_number=action.start_frame,
                    screenshot_path=action_frame.image_path if action_frame else None
                )
                
                workflow_steps.append(step)
            
            avg_confidence = sum(step.confidence_score for step in workflow_steps) / len(workflow_steps)
            estimated_duration = video_metadata.duration if video_metadata else 0.0
            
            workflow = WorkflowDefinition(
                name="Video Analysis Workflow",
                description=f"Workflow with {len(workflow_steps)} steps extracted from video analysis",
                steps=workflow_steps,
                estimated_duration=estimated_duration,
                confidence_score=avg_confidence,
                metadata={
                    'video_file': str(video_metadata.file_path) if video_metadata else None,
                    'total_frames': len(frames),
                }
            )
            
            logger.info(f"Generated workflow with {len(workflow_steps)} steps")
            return workflow
            
        except Exception as e:
            logger.error(f"Error generating workflow: {e}")
            return None


class FrameExtractionService:
    """Service for extracting frames from videos."""
    
    def __init__(self, config: VideoAnalysisConfig):
        self.config = config
        logger.info("FrameExtractionService initialized")
    
    async def extract_video_metadata(self, video_path: Path) -> VideoMetadata:
        """Extract metadata from video file."""
        # Mock implementation - replace with actual video processing
        return VideoMetadata(
            file_path=video_path,
            duration=30.0,
            fps=30.0,
            width=1920,
            height=1080,
            total_frames=900,
            format="mp4",
            size_bytes=video_path.stat().st_size if video_path.exists() else 0
        )
    
    async def extract_frames(self, video_path: Path) -> List[Frame]:
        """Extract frames from video file."""
        import numpy as np
        
        # Mock implementation - replace with actual frame extraction
        frames = []
        for i in range(min(10, self.config.max_frames)):  # Limit for demo
            frame = Frame(
                frame_number=i,
                timestamp=i * (1.0 / self.config.frame_extraction_fps),
                image=np.zeros((480, 640, 3), dtype=np.uint8),  # Mock image
                is_keyframe=(i % 5 == 0),
                visual_diff_score=0.1
            )
            frames.append(frame)
        
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames


class GeminiAnalysisService:
    """Service for Gemini-based video analysis."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.llm = None
        if self.api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=self.api_key,
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
        logger.info("GeminiAnalysisService initialized")
    
    async def analyze_video(
        self, 
        video_path: Path, 
        analysis_id: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze video using Gemini AI."""
        try:
            logger.info(f"Starting Gemini analysis for: {video_path}")
            
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            if not self.llm and not api_key:
                raise ValueError("Gemini API key not found")
            
            # Use provided API key if available
            if api_key and not self.llm:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    google_api_key=api_key,
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                )
            
            # Read and encode video
            with open(video_path, "rb") as video_file:
                encoded_video = base64.b64encode(video_file.read()).decode("utf-8")
            
            mime_type = self._get_mime_type(video_path)
            
            # Create message with video content
            message = HumanMessage(
                content=[
                    {"type": "text", "text": GEMINI_VIDEO_ANALYSIS_PROMPT},
                    {
                        "type": "media",
                        "data": encoded_video,
                        "mime_type": mime_type,
                    },
                ]
            )
            
            # Generate analysis
            response = self.llm.invoke([message])
            
            return {
                "success": True,
                "analysis": response.content,
                "model": "gemini-1.5-pro",
                "video_path": str(video_path),
                "analysis_id": analysis_id
            }
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "video_path": str(video_path),
                "analysis_id": analysis_id
            }
    
    def _get_mime_type(self, video_path: Path) -> str:
        """Get MIME type for video file based on extension."""
        mime_type_map = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
        }
        return mime_type_map.get(video_path.suffix.lower(), 'video/mp4') 