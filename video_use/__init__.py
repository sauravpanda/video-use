"""Video Use: Convert videos to browser-use workflows."""

__version__ = "0.1.0"

# Import from new modular structure
from .models import (
    VideoAnalysisResult, StructuredWorkflowOutput,
    TokenUsage, VideoAnalysisResponse, WorkflowExecutionResponse
)
from .services import VideoUseService, WorkflowGenerationService, WorkflowExecutionService
from .analysis import VideoAnalysisService, GeminiAnalysisService, FrameExtractionService
from .config import VideoAnalysisConfig

__all__ = [
    # Main services
    "VideoUseService",
    "WorkflowGenerationService",
    "WorkflowExecutionService",
    
    # Analysis services
    "VideoAnalysisService",
    "GeminiAnalysisService", 
    "FrameExtractionService",
    
    # Models
    "VideoAnalysisResult",
    "VideoAnalysisConfig",
    "StructuredWorkflowOutput",
    "TokenUsage",
    "VideoAnalysisResponse",
    "WorkflowExecutionResponse",
] 