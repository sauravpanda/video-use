"""Workflow generation service."""

import logging
import os
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from ..models.workflow import StructuredWorkflowOutput, TokenUsage
from ..config import VideoAnalysisConfig
from ..prompts import STRUCTURED_WORKFLOW_PROMPT, GEMINI_TO_STRUCTURED_PROMPT

logger = logging.getLogger(__name__)


class WorkflowGenerationService:
    """Service for generating structured workflows from video analysis."""
    
    def __init__(self, config: VideoAnalysisConfig):
        self.config = config
        logger.info("WorkflowGenerationService initialized")
        api_key = os.getenv('GOOGLE_API_KEY')
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=api_key,
            temperature=0.1,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    
    async def convert_actions_to_structured_output(
        self, actions, start_url: Optional[str] = None
    ) -> StructuredWorkflowOutput:
        """Convert extracted actions into structured output format using LLM."""
        from ..models.video import Action  # Avoid circular import
        
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
            
            response = await self.llm.ainvoke(messages)
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
        
        parser = PydanticOutputParser(pydantic_object=StructuredWorkflowOutput)
        
        # Create prompts
        human_prompt = f"""
Convert the following video analysis into a structured workflow output:

Analysis Text:
{gemini_analysis_text}

IMPORTANT - Starting URL: {start_url or 'Please infer from the analysis text'}
{"If a specific starting URL is provided above, you MUST use it exactly as given. Do not change or infer a different URL." if start_url else "Please infer the starting URL from the analysis text."}

Please extract:
1. Clear step-by-step automation instructions
2. Use the exact starting URL provided above (if given)
3. All specific data values (text inputs, form data, etc.) as parameters
"""
        
        try:
            messages = [
                SystemMessage(content=GEMINI_TO_STRUCTURED_PROMPT.format(
                    format_instructions=parser.get_format_instructions()
                )),
                HumanMessage(content=human_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
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
            
            formatted_actions.append(action_str)
        
        return '\n'.join(formatted_actions)
    
    def _extract_token_usage(self, response) -> TokenUsage:
        """Extract token usage from LLM response."""
        token_usage = TokenUsage()
        
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            token_usage.add_usage(
                input_tokens=getattr(usage, 'input_tokens', 0),
                output_tokens=getattr(usage, 'output_tokens', 0),
                model_name="gemini-1.5-pro"
            )
        elif hasattr(response, 'response_metadata'):
            metadata = response.response_metadata
            if 'usage' in metadata:
                usage = metadata['usage']
                token_usage.add_usage(
                    input_tokens=usage.get('input_tokens', 0),
                    output_tokens=usage.get('output_tokens', 0),
                    model_name="gemini-1.5-pro"
                )
        
        return token_usage 