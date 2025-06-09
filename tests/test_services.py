"""
Minimal tests for video-use services.
Clean and focused test implementation.
"""

import pytest

from video_use import (
    VideoUseService, VideoAnalysisConfig, 
    WorkflowGenerationService, TokenUsage
)
from video_use.models import Action, ActionType


class TestVideoUseService:
    """Test the main VideoUseService."""
    
    def test_initialization(self):
        """Test service initialization."""
        service = VideoUseService()
        assert service.config is not None
        assert service.analysis_service is not None
        assert service.workflow_service is not None
    
    def test_custom_config(self):
        """Test service with custom configuration."""
        config = VideoAnalysisConfig(
            frame_extraction_fps=2.0,
            max_frames=50
        )
        service = VideoUseService(config)
        assert service.config.frame_extraction_fps == 2.0
        assert service.config.max_frames == 50


class TestWorkflowGenerationService:
    """Test the WorkflowGenerationService."""
    
    def test_initialization(self):
        """Test service initialization."""
        config = VideoAnalysisConfig()
        service = WorkflowGenerationService(config)
        assert service.config is not None
    
    @pytest.mark.asyncio
    async def test_empty_actions(self):
        """Test with empty actions list."""
        config = VideoAnalysisConfig()
        service = WorkflowGenerationService(config)
        
        result = await service.convert_actions_to_structured_output([])
        
        assert result.prompt == 'No actions detected in video'
        assert result.start_url == ''
        assert result.parameters == {}
        assert isinstance(result.token_usage, TokenUsage)
    
    @pytest.mark.asyncio
    async def test_mock_actions(self):
        """Test with mock actions."""
        config = VideoAnalysisConfig()
        service = WorkflowGenerationService(config)
        
        # Create mock actions
        actions = [
            Action(
                action_type=ActionType.CLICK,
                start_frame=0,
                timestamp=0.0,
                confidence=0.8,
                description="Test click action"
            ),
            Action(
                action_type=ActionType.TYPE,
                start_frame=1,
                timestamp=1.0,
                confidence=0.9,
                description="Test type action",
                value="test input"
            )
        ]
        
        # This will fall back to basic structure without API key
        result = await service.convert_actions_to_structured_output(
            actions, 
            start_url="https://test.com"
        )
        
        assert "website" in result.prompt and "Click" in result.prompt
        assert result.start_url == "https://test.com"
        assert isinstance(result.token_usage, TokenUsage)


class TestTokenUsage:
    """Test TokenUsage model."""
    
    def test_initialization(self):
        """Test TokenUsage initialization."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.call_count == 0
    
    def test_add_usage(self):
        """Test adding token usage."""
        usage = TokenUsage()
        usage.add_usage(100, 50, "test-model")
        
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.call_count == 1
        assert usage.model_name == "test-model"
    
    def test_merge_usage(self):
        """Test merging token usage."""
        usage1 = TokenUsage()
        usage1.add_usage(100, 50, "model1")
        
        usage2 = TokenUsage()
        usage2.add_usage(200, 75, "model2")
        
        usage1.merge(usage2)
        
        assert usage1.input_tokens == 300
        assert usage1.output_tokens == 125
        assert usage1.total_tokens == 425
        assert usage1.call_count == 2


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"]) 