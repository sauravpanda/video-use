"""High-level video analysis service."""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import uuid
from datetime import datetime

from ..schema.models import (
    VideoAnalysisResult, VideoAnalysisConfig, 
    VideoUploadRequest, VideoAnalysisResponse
)
from .analyzer import VideoAnalyzer

logger = logging.getLogger(__name__)


class VideoService:
    """High-level service for video analysis operations."""
    
    def __init__(self, config: Optional[VideoAnalysisConfig] = None):
        self.config = config or VideoAnalysisConfig()
        self.analyzer = VideoAnalyzer(self.config)
        self.analysis_cache: Dict[str, VideoAnalysisResult] = {}
        
        logger.info("VideoService initialized")
    
    async def analyze_video_file(
        self, 
        video_path: Path, 
        analysis_id: Optional[str] = None,
        user_prompt: Optional[str] = None
    ) -> VideoAnalysisResponse:
        """
        Analyze a video file and return results.
        
        Args:
            video_path: Path to video file
            analysis_id: Optional analysis ID for tracking
            user_prompt: Optional user prompt for context
            
        Returns:
            VideoAnalysisResponse with results
        """
        # Generate analysis ID if not provided
        if not analysis_id:
            analysis_id = str(uuid.uuid4())
        
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting video analysis {analysis_id} for: {video_path}")
            
            # Perform analysis
            if user_prompt:
                result = await self.analyzer.analyze_video_with_prompt(video_path, user_prompt)
            else:
                result = await self.analyzer.analyze_video(video_path)
            
            # Cache result
            self.analysis_cache[analysis_id] = result
            
            # Convert to response format
            response = self._create_analysis_response(analysis_id, result)
            
            logger.info(f"Analysis {analysis_id} completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Analysis {analysis_id} failed: {e}")
            
            return VideoAnalysisResponse(
                success=False,
                analysis_id=analysis_id,
                workflow_steps=[],
                confidence_score=0.0,
                processing_time=0.0,
                error_message=str(e)
            )
    
    async def analyze_video_from_request(self, request: VideoUploadRequest) -> VideoAnalysisResponse:
        """Analyze video from upload request."""
        if not request.file_path:
            return VideoAnalysisResponse(
                success=False,
                analysis_id="",
                workflow_steps=[],
                confidence_score=0.0,
                processing_time=0.0,
                error_message="No file path provided"
            )
        
        video_path = Path(request.file_path)
        
        # Update config if provided
        if request.config:
            config = self._update_config_from_dict(request.config)
            analyzer = VideoAnalyzer(config)
        else:
            analyzer = self.analyzer
        
        # Perform analysis
        analysis_id = str(uuid.uuid4())
        result = await analyzer.analyze_video(video_path)
        
        # Cache result
        self.analysis_cache[analysis_id] = result
        
        return self._create_analysis_response(analysis_id, result)
    
    def _create_analysis_response(self, analysis_id: str, result: VideoAnalysisResult) -> VideoAnalysisResponse:
        """Convert analysis result to response format."""
        workflow_steps = []
        
        if result.workflow and result.workflow.steps:
            for step in result.workflow.steps:
                step_dict = {
                    'step_id': step.step_id,
                    'action_type': step.action.action_type.value,
                    'description': step.description,
                    'timestamp': step.video_timestamp,
                    'confidence': step.confidence_score,
                    'frame_number': step.frame_number
                }
                
                # Add target element info if available
                if step.action.target_element:
                    step_dict['target_element'] = {
                        'type': step.action.target_element.element_type.value,
                        'text': step.action.target_element.text,
                        'bbox': step.action.target_element.bbox,
                        'confidence': step.action.target_element.confidence
                    }
                
                # Add action value if available
                if step.action.value:
                    step_dict['value'] = step.action.value
                
                # Add coordinates if available
                if step.action.coordinates:
                    step_dict['coordinates'] = step.action.coordinates
                
                workflow_steps.append(step_dict)
        
        return VideoAnalysisResponse(
            success=result.success,
            analysis_id=analysis_id,
            workflow_steps=workflow_steps,
            confidence_score=result.workflow.confidence_score if result.workflow else 0.0,
            processing_time=result.processing_time,
            error_message=result.error_message
        )
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]) -> VideoAnalysisConfig:
        """Update configuration from dictionary."""
        config = VideoAnalysisConfig()
        
        # Update config fields if provided
        if 'frame_extraction_fps' in config_dict:
            config.frame_extraction_fps = config_dict['frame_extraction_fps']
        
        if 'ui_detection_confidence' in config_dict:
            config.ui_detection_confidence = config_dict['ui_detection_confidence']
        
        if 'action_confidence_threshold' in config_dict:
            config.action_confidence_threshold = config_dict['action_confidence_threshold']
        
        if 'enable_ocr' in config_dict:
            config.enable_ocr = config_dict['enable_ocr']
        
        if 'llm_model' in config_dict:
            config.llm_model = config_dict['llm_model']
        
        if 'max_frames' in config_dict:
            config.max_frames = config_dict['max_frames']
        
        if 'parallel_processing' in config_dict:
            config.parallel_processing = config_dict['parallel_processing']
        
        return config
    
    async def get_analysis_result(self, analysis_id: str) -> Optional[VideoAnalysisResult]:
        """Get cached analysis result by ID."""
        return self.analysis_cache.get(analysis_id)
    
    async def list_analyses(self) -> List[Dict[str, Any]]:
        """List all cached analyses."""
        analyses = []
        
        for analysis_id, result in self.analysis_cache.items():
            analysis_info = {
                'analysis_id': analysis_id,
                'success': result.success,
                'processing_time': result.processing_time,
                'actions_count': len(result.actions),
                'workflow_name': result.workflow.name if result.workflow else None,
                'video_file': str(result.video_metadata.file_path) if result.video_metadata else None
            }
            analyses.append(analysis_info)
        
        return analyses
    
    async def export_workflow_to_browser_use(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Export workflow to browser-use compatible format.
        
        This method converts the video analysis workflow into a format
        that can be executed by the browser-use library.
        """
        result = await self.get_analysis_result(analysis_id)
        
        if not result or not result.workflow:
            return None
        
        # Convert to browser-use format
        browser_use_workflow = {
            'name': result.workflow.name,
            'description': result.workflow.description,
            'steps': []
        }
        
        for step in result.workflow.steps:
            action = step.action
            
            # Convert action to browser-use command
            browser_use_step = {
                'step_id': step.step_id,
                'description': step.description,
                'command': self._convert_to_browser_use_command(action),
                'confidence': step.confidence_score,
                'timestamp': step.video_timestamp
            }
            
            browser_use_workflow['steps'].append(browser_use_step)
        
        return browser_use_workflow
    
    def _convert_to_browser_use_command(self, action) -> Dict[str, Any]:
        """Convert video action to browser-use command."""
        command = {
            'type': action.action_type.value
        }
        
        if action.action_type.value == 'click':
            if action.target_element:
                if action.target_element.text:
                    command['selector'] = f"text={action.target_element.text}"
                else:
                    # Use coordinates if no text available
                    command['coordinates'] = action.coordinates
            elif action.coordinates:
                command['coordinates'] = action.coordinates
        
        elif action.action_type.value == 'type':
            command['text'] = action.value
            if action.target_element and action.target_element.text:
                command['selector'] = f"text={action.target_element.text}"
        
        elif action.action_type.value == 'scroll':
            command['direction'] = action.value or 'down'
        
        elif action.action_type.value == 'navigate':
            command['url'] = action.value if action.value else 'detected_navigation'
        
        return command
    
    async def quick_analysis(self, video_path: Path) -> VideoAnalysisResponse:
        """
        Perform quick analysis using keyframes only.
        
        This is faster but less accurate than full analysis.
        """
        analysis_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Starting quick analysis {analysis_id} for: {video_path}")
            
            result = await self.analyzer.extract_keyframe_actions(video_path)
            
            # Cache result
            self.analysis_cache[analysis_id] = result
            
            response = self._create_analysis_response(analysis_id, result)
            response.analysis_id = f"{analysis_id}_quick"
            
            logger.info(f"Quick analysis {analysis_id} completed")
            return response
            
        except Exception as e:
            logger.error(f"Quick analysis {analysis_id} failed: {e}")
            
            return VideoAnalysisResponse(
                success=False,
                analysis_id=f"{analysis_id}_quick",
                workflow_steps=[],
                confidence_score=0.0,
                processing_time=0.0,
                error_message=str(e)
            )
    
    async def save_analysis(self, analysis_id: str, output_dir: Path) -> Optional[Path]:
        """Save analysis results to disk."""
        result = await self.get_analysis_result(analysis_id)
        
        if not result:
            logger.error(f"Analysis {analysis_id} not found")
            return None
        
        try:
            results_file = await self.analyzer.save_analysis_results(result, output_dir)
            logger.info(f"Analysis {analysis_id} saved to {output_dir}")
            return results_file
            
        except Exception as e:
            logger.error(f"Failed to save analysis {analysis_id}: {e}")
            return None
    
    async def cleanup_cache(self, max_age_hours: int = 24) -> int:
        """Clean up old analysis results from cache."""
        # For now, just clear all cache
        # In a real implementation, you'd check timestamps
        count = len(self.analysis_cache)
        self.analysis_cache.clear()
        
        logger.info(f"Cleaned up {count} cached analyses")
        return count
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported video formats."""
        return ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
    
    def validate_video_file(self, video_path: Path) -> bool:
        """Validate if video file is supported."""
        if not video_path.exists():
            return False
        
        if video_path.suffix.lower() not in self.get_supported_formats():
            return False
        
        # Additional validation could be added here
        return True 