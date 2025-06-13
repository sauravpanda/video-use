"""Default configuration values."""

from .analysis import VideoAnalysisConfig

# Default configuration instance
DEFAULT_CONFIG = VideoAnalysisConfig()

# Environment-specific configurations
PRODUCTION_CONFIG = VideoAnalysisConfig(
    parallel_processing=True,
    max_workers=8,
    enable_caching=True,
    frame_extraction_fps=2.0,
    max_frames=500
)

DEVELOPMENT_CONFIG = VideoAnalysisConfig(
    parallel_processing=False,
    max_workers=2,
    enable_caching=False,
    frame_extraction_fps=1.0,
    max_frames=100,
    generate_descriptions=True
)

TESTING_CONFIG = VideoAnalysisConfig(
    parallel_processing=False,
    max_workers=1,
    enable_caching=False,
    frame_extraction_fps=0.5,
    max_frames=20,
    generate_descriptions=False,
    include_validation_rules=False
) 