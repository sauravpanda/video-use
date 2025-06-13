"""Main video analysis service."""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from ..models.video import VideoAnalysisResult
from ..models.workflow import StructuredWorkflowOutput
from ..models.api import VideoAnalysisResponse, WorkflowExecutionResponse
from ..config import VideoAnalysisConfig
from ..analysis.services import VideoAnalysisService, GeminiAnalysisService
from .workflow_service import WorkflowGenerationService
from .execution_service import WorkflowExecutionService

logger = logging.getLogger(__name__)


class VideoUseService:
    """High-level service for video analysis operations."""
    
    def __init__(self, config: Optional[VideoAnalysisConfig] = None):
        self.config = config or VideoAnalysisConfig()
        self.analysis_service = VideoAnalysisService(self.config)
        self.gemini_service = GeminiAnalysisService()
        self.workflow_service = WorkflowGenerationService(self.config)
        self.execution_service = WorkflowExecutionService()
        self.analysis_cache: Dict[str, VideoAnalysisResult] = {}
        
        logger.info("VideoUseService initialized")
    
    async def analyze_video_file(
        self, 
        video_path: Path, 
        analysis_id: Optional[str] = None,
        user_prompt: Optional[str] = None,
        use_gemini: bool = False,
        gemini_api_key: Optional[str] = None
    ) -> VideoAnalysisResponse:
        """
        Analyze a video file and return results.
        
        Args:
            video_path: Path to video file
            analysis_id: Optional analysis ID for tracking
            user_prompt: Optional user prompt for context
            use_gemini: Whether to use Gemini AI analysis instead of traditional analysis
            gemini_api_key: Optional Gemini API key
            
        Returns:
            VideoAnalysisResponse with results
        """
        if not analysis_id:
            analysis_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Starting video analysis {analysis_id} for: {video_path}")
            
            if use_gemini:
                # Use Gemini AI analysis
                gemini_result = await self.gemini_service.analyze_video(
                    video_path, analysis_id, gemini_api_key
                )
                
                if gemini_result["success"]:
                    workflow_steps = [{
                        "step_id": "gemini_analysis",
                        "action_type": "analysis",
                        "description": "AI-generated video analysis",
                        "analysis_text": gemini_result["analysis"],
                        "model": gemini_result["model"],
                        "timestamp": 0.0,
                        "confidence": 1.0,
                        "frame_number": 0
                    }]
                    
                    return VideoAnalysisResponse(
                        success=True,
                        analysis_id=analysis_id,
                        workflow_steps=workflow_steps,
                        confidence_score=1.0,
                        processing_time=0.0
                    )
                else:
                    return VideoAnalysisResponse(
                        success=False,
                        analysis_id=analysis_id,
                        workflow_steps=[],
                        confidence_score=0.0,
                        processing_time=0.0,
                        error_message=gemini_result["error"]
                    )
            else:
                # Use traditional frame-based analysis
                if user_prompt:
                    result = await self.analysis_service.analyze_video_with_prompt(
                        video_path, user_prompt
                    )
                else:
                    result = await self.analysis_service.analyze_video(video_path)
                
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
    
    async def generate_structured_workflow(
        self, 
        analysis_id: str, 
        start_url: Optional[str] = None
    ) -> StructuredWorkflowOutput:
        """Generate structured workflow from cached analysis result."""
        if analysis_id not in self.analysis_cache:
            from ..models.workflow import TokenUsage
            return StructuredWorkflowOutput(
                prompt="Analysis not found",
                start_url=start_url or "",
                parameters={},
                token_usage=TokenUsage()
            )
        
        result = self.analysis_cache[analysis_id]
        return await self.workflow_service.convert_actions_to_structured_output(
            result.actions, start_url
        )
    
    async def generate_structured_workflow_from_gemini(
        self,
        gemini_analysis_text: str,
        start_url: Optional[str] = None
    ) -> StructuredWorkflowOutput:
        """Generate structured workflow from Gemini analysis text."""
        return await self.workflow_service.convert_gemini_analysis_to_structured_output(
            gemini_analysis_text, start_url
        )
    
    async def execute_workflow(
        self,
        workflow: StructuredWorkflowOutput,
        execution_id: Optional[str] = None,
        headless: bool = False,
        timeout: int = 30,
        use_shared_session: bool = True
    ) -> WorkflowExecutionResponse:
        """
        Execute a generated workflow using browser-use agent.
        
        Args:
            workflow: The structured workflow to execute
            execution_id: Optional execution ID for tracking
            headless: Whether to run browser in headless mode
            timeout: Timeout in seconds for workflow execution
            use_shared_session: Whether to use a shared browser session
            
        Returns:
            WorkflowExecutionResponse with execution results
        """
        if not self.execution_service:
            return WorkflowExecutionResponse(
                success=False,
                execution_id=execution_id or "unknown",
                results=[],
                execution_time=0.0,
                error_message="Browser-use not available for workflow execution"
            )
        
        return await self.execution_service.execute_workflow(
            workflow, execution_id, headless, timeout, use_shared_session
        )
    
    async def analyze_and_execute_workflow(
        self,
        video_path: Path,
        start_url: Optional[str] = None,
        use_gemini: bool = True,
        gemini_api_key: Optional[str] = None,
        headless: bool = False,
        timeout: int = 30,
        use_shared_session: bool = True
    ) -> Dict[str, Any]:
        """
        Complete pipeline: analyze video and execute the resulting workflow.
        
        Args:
            video_path: Path to video file
            start_url: Starting URL for workflow execution
            use_gemini: Whether to use Gemini for analysis
            gemini_api_key: Optional Gemini API key
            headless: Whether to run browser in headless mode
            timeout: Timeout for workflow execution
            use_shared_session: Whether to use a shared browser session
            
        Returns:
            Dictionary containing analysis and execution results
        """
        # Step 1: Analyze video
        analysis_result = await self.analyze_video_file(
            video_path, 
            use_gemini=use_gemini, 
            gemini_api_key=gemini_api_key
        )
        
        if not analysis_result.success:
            return {
                "success": False,
                "error": f"Video analysis failed: {analysis_result.error_message}",
                "analysis_result": analysis_result,
                "workflow": None,
                "execution_result": None
            }
        
        # Step 2: Generate structured workflow
        try:
            if use_gemini and analysis_result.workflow_steps:
                # Extract analysis text from Gemini result
                analysis_text = analysis_result.workflow_steps[0].get('analysis_text', '')
                workflow = await self.generate_structured_workflow_from_gemini(
                    analysis_text, start_url
                )
            else:
                workflow = await self.generate_structured_workflow(
                    analysis_result.analysis_id, start_url
                )
        except Exception as e:
            return {
                "success": False,
                "error": f"Workflow generation failed: {e}",
                "analysis_result": analysis_result,
                "workflow": None,
                "execution_result": None
            }
        
        # Step 3: Execute workflow
        execution_result = await self.execute_workflow(
            workflow, 
            headless=headless, 
            timeout=timeout,
            use_shared_session=use_shared_session
        )
        
        return {
            "success": execution_result.success,
            "analysis_result": analysis_result,
            "workflow": workflow,
            "execution_result": execution_result,
            "error": execution_result.error_message if not execution_result.success else None
        }
    
    def _create_analysis_response(
        self, analysis_id: str, result: VideoAnalysisResult
    ) -> VideoAnalysisResponse:
        """Convert VideoAnalysisResult to VideoAnalysisResponse."""
        workflow_steps = []
        
        # Convert actions to workflow steps
        for i, action in enumerate(result.actions):
            step = {
                "step_id": f"step_{i+1}",
                "action_type": action.action_type.value if hasattr(action.action_type, 'value') else str(action.action_type),
                "description": action.description,
                "timestamp": action.timestamp,
                "confidence": action.confidence,
                "frame_number": getattr(action, 'start_frame', 0),
                "target_element": None
            }
            
            if action.target_element:
                step["target_element"] = {
                    "type": action.target_element.element_type.value if hasattr(action.target_element.element_type, 'value') else str(action.target_element.element_type),
                    "text": action.target_element.text,
                    "bbox": action.target_element.bbox,
                    "confidence": action.target_element.confidence
                }
            
            workflow_steps.append(step)
        
        return VideoAnalysisResponse(
            success=result.success,
            analysis_id=analysis_id,
            workflow_steps=workflow_steps,
            confidence_score=result.workflow.confidence_score if result.workflow else 0.0,
            processing_time=result.processing_time,
            error_message=result.error_message
        ) 