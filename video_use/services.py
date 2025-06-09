"""Main business logic services for video-use."""

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from .models import (
    VideoAnalysisResult, VideoAnalysisConfig, StructuredWorkflowOutput,
    TokenUsage, VideoAnalysisResponse
)
from .prompts import STRUCTURED_WORKFLOW_PROMPT, GEMINI_TO_STRUCTURED_PROMPT
from .analysis.services import VideoAnalysisService, GeminiAnalysisService

logger = logging.getLogger(__name__)


class VideoUseService:
    """High-level service for video analysis operations."""
    
    def __init__(self, config: Optional[VideoAnalysisConfig] = None):
        self.config = config or VideoAnalysisConfig()
        self.analysis_service = VideoAnalysisService(self.config)
        self.gemini_service = GeminiAnalysisService()
        self.workflow_service = WorkflowGenerationService(self.config)
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
    
    def _create_analysis_response(
        self, analysis_id: str, result: VideoAnalysisResult
    ) -> VideoAnalysisResponse:
        """Convert VideoAnalysisResult to VideoAnalysisResponse."""
        workflow_steps = []
        
        if result.workflow and result.workflow.steps:
            for step in result.workflow.steps:
                workflow_steps.append({
                    "step_id": step.step_id,
                    "action_type": step.action.action_type.value if hasattr(step.action.action_type, 'value') else str(step.action.action_type),
                    "description": step.description,
                    "timestamp": step.video_timestamp,
                    "confidence": step.confidence_score,
                    "frame_number": step.frame_number
                })
        
        return VideoAnalysisResponse(
            success=result.success,
            analysis_id=analysis_id,
            workflow_steps=workflow_steps,
            confidence_score=result.workflow.confidence_score if result.workflow else 0.0,
            processing_time=result.processing_time,
            error_message=result.error_message
        )


class WorkflowGenerationService:
    """Service for generating structured workflows from video analysis."""
    
    def __init__(self, config: VideoAnalysisConfig):
        self.config = config
        logger.info("WorkflowGenerationService initialized")
    
    async def convert_actions_to_structured_output(
        self, actions, start_url: Optional[str] = None
    ) -> StructuredWorkflowOutput:
        """Convert extracted actions into structured output format using LLM."""
        from .models import Action  # Avoid circular import
        
        if not actions:
            return StructuredWorkflowOutput(
                prompt='No actions detected in video',
                start_url=start_url or '',
                parameters={},
                token_usage=TokenUsage()
            )
        
        # Initialize Gemini LLM
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            logger.warning("GOOGLE_API_KEY not found, falling back to basic structure")
            return StructuredWorkflowOutput(
                prompt=f"Workflow with {len(actions)} actions detected from video analysis",
                start_url=start_url or '',
                parameters={},
                token_usage=TokenUsage()
            )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=api_key,
            temperature=0.1,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        parser = PydanticOutputParser(pydantic_object=StructuredWorkflowOutput)
        
        # Prepare actions data for LLM
        actions_data = self._prepare_actions_data(actions)
        
        # Create prompts
        human_prompt = f"""
Based on the following actions detected from video analysis, create a structured workflow output:

Actions detected:
{self._format_actions_for_llm(actions_data)}

Starting URL hint: {start_url or 'Not provided - please infer from navigation actions'}

Please generate clear workflow instructions and extract all relevant parameters.
"""
        
        try:
            messages = [
                SystemMessage(content=STRUCTURED_WORKFLOW_PROMPT.format(
                    format_instructions=parser.get_format_instructions()
                )),
                HumanMessage(content=human_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            result = parser.parse(response.content)
            
            # Track token usage
            token_usage = self._extract_token_usage(response)
            result.token_usage = token_usage
            
            logger.info(f"Generated structured workflow with {len(result.parameters)} parameters (tokens: {token_usage.total_tokens})")
            return result
            
        except Exception as e:
            logger.error(f"Error generating structured output with LLM: {e}")
            return StructuredWorkflowOutput(
                prompt=f"Workflow with {len(actions)} actions detected from video analysis",
                start_url=start_url or '',
                parameters={},
                token_usage=TokenUsage()
            )
    
    async def convert_gemini_analysis_to_structured_output(
        self, 
        gemini_analysis_text: str, 
        start_url: Optional[str] = None
    ) -> StructuredWorkflowOutput:
        """Convert Gemini analysis text into structured output format using LLM."""
        if not gemini_analysis_text.strip():
            return StructuredWorkflowOutput(
                prompt='No analysis text provided',
                start_url=start_url or '',
                parameters={},
                token_usage=TokenUsage()
            )
        
        # Initialize Gemini LLM
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            logger.warning("GOOGLE_API_KEY not found, falling back to basic structure")
            return StructuredWorkflowOutput(
                prompt=f"Manual workflow based on analysis: {gemini_analysis_text[:200]}...",
                start_url=start_url or '',
                parameters={},
                token_usage=TokenUsage()
            )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=api_key,
            temperature=0.1,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        parser = PydanticOutputParser(pydantic_object=StructuredWorkflowOutput)
        
        # Create prompts
        human_prompt = f"""
Convert the following video analysis into a structured workflow output:

Analysis Text:
{gemini_analysis_text}

Starting URL hint: {start_url or 'Please infer from the analysis text'}

Please extract:
1. Clear step-by-step automation instructions
2. Starting URL mentioned in the analysis
3. All specific data values (text inputs, form data, etc.) as parameters
"""
        
        try:
            messages = [
                SystemMessage(content=GEMINI_TO_STRUCTURED_PROMPT.format(
                    format_instructions=parser.get_format_instructions()
                )),
                HumanMessage(content=human_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            result = parser.parse(response.content)
            
            # Track token usage
            token_usage = self._extract_token_usage(response)
            result.token_usage = token_usage
            
            logger.info(f"Generated structured workflow from Gemini analysis with {len(result.parameters)} parameters (tokens: {token_usage.total_tokens})")
            return result
            
        except Exception as e:
            logger.error(f"Error generating structured output from Gemini analysis: {e}")
            return StructuredWorkflowOutput(
                prompt=f"Manual workflow based on analysis: {gemini_analysis_text[:500]}...",
                start_url=start_url or '',
                parameters={},
                token_usage=TokenUsage()
            )
    
    def _prepare_actions_data(self, actions) -> list:
        """Prepare actions data for LLM processing."""
        actions_data = []
        for i, action in enumerate(actions):
            action_info = {
                'step': i + 1,
                'action_type': action.action_type.value if hasattr(action.action_type, 'value') else str(action.action_type),
                'timestamp': action.timestamp,
                'confidence': action.confidence,
                'description': action.description or '',
                'target_element': None,
                'value': getattr(action, 'value', None)
            }
            
            # Add target element info if available
            if hasattr(action, 'target_element') and action.target_element:
                element = action.target_element
                action_info['target_element'] = {
                    'type': element.element_type.value if hasattr(element.element_type, 'value') else str(element.element_type),
                    'text': getattr(element, 'text', ''),
                    'attributes': getattr(element, 'attributes', {}),
                    'xpath': getattr(element, 'xpath', ''),
                    'css_selector': getattr(element, 'css_selector', '')
                }
            
            actions_data.append(action_info)
        
        return actions_data
    
    def _format_actions_for_llm(self, actions_data: list) -> str:
        """Format actions data for LLM input."""
        formatted_actions = []
        
        for action in actions_data:
            action_str = f"Step {action['step']}: {action['action_type'].upper()}"
            
            if action['value']:
                action_str += f" (value: '{action['value']}')"
            
            if action['target_element']:
                element = action['target_element']
                if element['text']:
                    action_str += f" on element '{element['text']}'"
                elif element['type']:
                    action_str += f" on {element['type']}"
            
            if action['description']:
                action_str += f" - {action['description']}"
            
            action_str += f" (confidence: {action['confidence']:.2f}, timestamp: {action['timestamp']:.2f}s)"
            formatted_actions.append(action_str)
        
        return "\n".join(formatted_actions)
    
    def _extract_token_usage(self, response) -> TokenUsage:
        """Extract token usage from LLM response."""
        token_usage = TokenUsage()
        
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            input_tokens = getattr(usage, 'input_tokens', 0)
            output_tokens = getattr(usage, 'output_tokens', 0)
            token_usage.add_usage(input_tokens, output_tokens, "gemini-1.5-pro")
        elif hasattr(response, 'response_metadata') and response.response_metadata:
            usage = response.response_metadata.get('token_usage', {})
            input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
            token_usage.add_usage(input_tokens, output_tokens, "gemini-1.5-pro")
        else:
            # Estimate tokens if usage data not available
            input_estimate = 100  # Rough estimate
            output_estimate = 50
            token_usage.add_usage(input_estimate, output_estimate, "gemini-1.5-pro")
            logger.warning("Token usage not available from API, using estimated values")
        
        return token_usage 