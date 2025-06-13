"""Configuration for video analysis."""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class VideoAnalysisConfig:
    """Configuration for video analysis."""
    # Frame extraction
    frame_extraction_fps: float = 1.0
    min_frame_difference: float = 0.02
    max_frames: int = 1000
    
    # UI detection
    ui_detection_confidence: float = 0.7
    ocr_languages: List[str] = field(default_factory=lambda: ['en'])
    enable_ocr: bool = True
    
    # Action inference
    action_confidence_threshold: float = 0.6
    temporal_smoothing_window: int = 3
    enable_action_grouping: bool = True
    
    # Workflow generation
    llm_model: str = "gemini-1.5-pro"
    max_workflow_steps: int = 50
    generate_descriptions: bool = True
    include_validation_rules: bool = True
    
    # Performance
    parallel_processing: bool = True
    max_workers: int = 4
    enable_caching: bool = True
    cache_dir: Optional[Path] = None 