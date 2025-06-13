"""API request and response models."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class VideoUploadRequest(BaseModel):
    """Request model for video upload."""
    file_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class VideoAnalysisResponse(BaseModel):
    """Response model for video analysis."""
    success: bool
    analysis_id: str
    workflow_steps: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    error_message: Optional[str] = None


class WorkflowExecutionRequest(BaseModel):
    """Request model for workflow execution."""
    workflow_id: str
    variables: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None


class WorkflowExecutionResponse(BaseModel):
    """Response model for workflow execution."""
    success: bool
    execution_id: str
    results: List[Dict[str, Any]]
    execution_time: float
    error_message: Optional[str] = None 