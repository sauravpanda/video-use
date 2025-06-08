"""Core data models for video-use."""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Union
from pathlib import Path
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Types of browser actions that can be detected."""
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    NAVIGATE = "navigate"
    WAIT = "wait"
    SELECT = "select"
    DRAG_DROP = "drag_drop"
    HOVER = "hover"
    KEY_PRESS = "key_press"


class UIElementType(str, Enum):
    """Types of UI elements that can be detected."""
    BUTTON = "button"
    INPUT = "input"
    LINK = "link"
    TEXT = "text"
    IMAGE = "image"
    DROPDOWN = "dropdown"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    FORM = "form"
    MENU = "menu"
    DIALOG = "dialog"
    TAB = "tab"


@dataclass
class VideoMetadata:
    """Metadata about the input video."""
    file_path: Path
    duration: float
    fps: float
    width: int
    height: int
    total_frames: int
    format: str
    size_bytes: int


@dataclass
class Frame:
    """Represents a single video frame."""
    frame_number: int
    timestamp: float
    image: np.ndarray
    image_path: Optional[Path] = None
    is_keyframe: bool = False
    visual_diff_score: float = 0.0


@dataclass
class UIElement:
    """Represents a detected UI element in a frame."""
    element_type: UIElementType
    bbox: tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    text: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    frame_number: int = 0
    xpath: Optional[str] = None
    css_selector: Optional[str] = None


@dataclass
class Action:
    """Represents an inferred user action."""
    action_type: ActionType
    target_element: Optional[UIElement] = None
    value: Optional[str] = None
    start_frame: int = 0
    end_frame: int = 0
    timestamp: float = 0.0
    confidence: float = 0.0
    description: str = ""
    coordinates: Optional[tuple[int, int]] = None


@dataclass
class VideoWorkflowStep:
    """A workflow step derived from video analysis."""
    step_id: str
    action: Action
    description: str
    video_timestamp: float
    confidence_score: float
    frame_number: int
    screenshot_path: Optional[Path] = None
    dependencies: List[str] = field(default_factory=list)
    validation_rules: List[str] = field(default_factory=list)


@dataclass
class WorkflowDefinition:
    """Complete workflow definition generated from video."""
    name: str
    description: str
    steps: List[VideoWorkflowStep]
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: float = 0.0
    confidence_score: float = 0.0


@dataclass
class VideoAnalysisResult:
    """Result of video analysis containing all extracted information."""
    video_metadata: VideoMetadata
    frames: List[Frame]
    ui_elements: List[UIElement]
    actions: List[Action]
    workflow: Optional[WorkflowDefinition] = None
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if the analysis result is valid."""
        return self.success and self.error_message is None


@dataclass
class VideoAnalysisConfig:
    """Configuration for video analysis."""
    # Frame extraction
    frame_extraction_fps: float = 1.0  # Extract 1 frame per second
    min_frame_difference: float = 0.02  # Minimum visual change threshold (2% instead of 10%)
    max_frames: int = 1000  # Maximum number of frames to process
    
    # UI detection
    ui_detection_confidence: float = 0.7
    ocr_languages: List[str] = field(default_factory=lambda: ['en'])
    enable_ocr: bool = True
    
    # Action inference
    action_confidence_threshold: float = 0.6  # Lower threshold to detect more subtle actions
    temporal_smoothing_window: int = 3  # frames
    enable_action_grouping: bool = True
    
    # Workflow generation
    llm_model: str = "gpt-4o"
    max_workflow_steps: int = 50
    generate_descriptions: bool = False  # Disable LLM descriptions temporarily
    include_validation_rules: bool = True
    
    # Performance
    parallel_processing: bool = True
    max_workers: int = 4
    enable_caching: bool = True
    cache_dir: Optional[Path] = None


# Pydantic models for API endpoints
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