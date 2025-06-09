# Video-Use

**Convert browser interaction videos into automated workflows using AI and computer vision.**

Video-Use analyzes screen recordings of browser interactions and automatically generates executable workflows compatible with [browser-use](https://github.com/browser-use/browser-use) for automation.

## 🎯 What it does

1. **Extracts frames** from your browser interaction videos
2. **Analyzes user interactions** using AI (Gemini) to understand step-by-step actions
3. **Generates structured workflows** that can be executed with browser-use automation
4. **Provides both frame-based and AI-powered analysis** for different use cases

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sauravpanda/video-use.git
cd video-use

# Install dependencies
pip install -e .
```

### Basic Usage

#### Python API

```python
from video_use import VideoUseService, VideoAnalysisConfig
from pathlib import Path
import asyncio

async def main():
    # Initialize service
    config = VideoAnalysisConfig(
        frame_extraction_fps=1.0,
        max_frames=20
    )
    service = VideoUseService(config)
    
    # For Gemini AI analysis (requires GOOGLE_API_KEY)
    result = await service.analyze_video_file(
        Path("recording.mp4"),
        use_gemini=True
    )
    
    if result.success:
        # Generate structured workflow
        workflow = await service.generate_structured_workflow_from_gemini(
            result.workflow_steps[0]['analysis_text'],
            start_url="https://example.com"
        )
        print(f"Generated workflow: {workflow.prompt}")

asyncio.run(main())
```

#### Running Examples

```bash
# Basic frame extraction
cd examples
python simple_example.py sample_form_filling.mp4

# Frame extraction with more detail
cd examples/frame-extraction
python example_frame_extraction.py ../sample_form_filling.mp4 --mode frames

# AI analysis (requires GOOGLE_API_KEY environment variable)
export GOOGLE_API_KEY="your-gemini-api-key"
python example_frame_extraction.py ../sample_form_filling.mp4 --mode gemini
```

## 📋 Features

### Core Capabilities
- **Multi-format support**: MP4, AVI, MOV, MKV, WebM support for video input
- **Intelligent frame extraction**: Adaptive sampling based on visual changes
- **AI-powered analysis**: Uses Google Gemini for understanding user interactions
- **Structured workflow generation**: Converts analysis into browser-use compatible formats
- **Frame-based analysis**: Traditional computer vision approach for detailed frame inspection
- **Flexible configuration**: Customizable analysis parameters

### Analysis Features
- **Gemini AI integration**: Advanced natural language understanding of user actions
- **Frame extraction service**: Smart sampling to identify key interaction moments
- **Asynchronous processing**: Non-blocking analysis for better performance
- **Multiple analysis modes**: Choose between AI analysis or traditional frame processing
- **Workflow export**: Generate structured outputs compatible with browser automation

## 🛠️ Configuration

### Analysis Configuration

```python
from video_use import VideoAnalysisConfig

config = VideoAnalysisConfig(
    # Frame extraction settings
    frame_extraction_fps=1.0,           # Extract 1 frame per second
    min_frame_difference=0.02,           # Minimum difference to consider frames different
    max_frames=1000,                     # Maximum frames to process
    
    # AI analysis settings
    llm_model="gemini-1.5-pro",        # Gemini model for analysis
    
    # Performance settings
    parallel_processing=True,            # Enable parallel processing
    max_workers=4,                      # Number of worker threads
    enable_caching=True                 # Enable result caching
)
```

### Environment Variables

```bash
# Required for Gemini AI analysis
export GOOGLE_API_KEY="your-gemini-api-key"

# Optional: specify API endpoint
export GOOGLE_API_BASE="https://generativelanguage.googleapis.com"
```

## 📊 Example Output

### Frame Extraction Results
```bash
🔍 Extracting frames from: sample_form_filling.mp4
✅ Extracted 15 frames
   Frame 1: #30 at 1.00s
   Frame 2: #60 at 2.00s
   Frame 3: #90 at 3.00s
   Frame 4: #120 at 4.00s
   Frame 5: #150 at 5.00s
   ... and 10 more frames
```

### AI Analysis Results
```bash
🤖 Analyzing video with Gemini: sample_form_filling.mp4
✅ Analysis complete!
============================================================
STEP-BY-STEP USER ACTIONS:
============================================================

1. The user navigates to a login page at the beginning of the video
2. They click on the username/email input field
3. The user types their email address into the field
4. Next, they click on the password input field
5. The user enters their password
6. Finally, they click the "Login" or "Sign In" button to submit the form
7. The page transitions to show a successful login or dashboard

This appears to be a standard login workflow with form interaction.
```

### Structured Workflow Output
```python
workflow = StructuredWorkflowOutput(
    prompt="Navigate to login page, fill out username and password, then submit the form",
    start_url="https://example.com/login",
    parameters={
        "username": "user@example.com",
        "password": "[HIDDEN]",
        "login_button_text": "Login"
    },
    token_usage=TokenUsage(
        input_tokens=1250,
        output_tokens=180,
        total_tokens=1430
    )
)
```

## 🎥 Recording Tips

For best analysis results:

### Recording Setup
- **Resolution**: Use 1080p or higher
- **Frame rate**: 30fps or higher recommended
- **Browser**: Full screen or consistent window size
- **Clean UI**: Avoid overlapping windows or notifications

### Interaction Guidelines
- **Deliberate movements**: Move mouse smoothly and deliberately
- **Clear clicks**: Click precisely on target elements
- **Pause between actions**: Brief pause after each action helps detection
- **Visible UI**: Ensure buttons and inputs are clearly visible
- **Text input**: Type at normal speed, avoid very fast typing

### What Works Best
- ✅ Form filling workflows
- ✅ Button click sequences  
- ✅ Navigation flows
- ✅ Multi-step processes
- ✅ E-commerce interactions

### Limitations
- ❌ Very fast mouse movements
- ❌ Drag and drop (limited support)
- ❌ Keyboard shortcuts
- ❌ Complex animations

## 🔧 Available Examples

### Frame Extraction Example
```bash
cd examples/frame-extraction
python example_frame_extraction.py ../sample_form_filling.mp4 --mode frames
```

### AI Analysis Example  
```bash
# Set up Gemini API key
export GOOGLE_API_KEY="your-gemini-api-key"

# Run AI analysis
cd examples/frame-extraction
python example_frame_extraction.py ../sample_form_filling.mp4 --mode gemini
```

### Simple Service Example
```bash
cd examples
python simple_example.py sample_form_filling.mp4
```

### Custom Python Integration
```python
from video_use import VideoUseService, VideoAnalysisConfig
from pathlib import Path

# Create service with custom config
config = VideoAnalysisConfig(
    frame_extraction_fps=2.0,
    max_frames=50
)
service = VideoUseService(config)

# Analyze video
result = await service.analyze_video_file(
    Path("your_video.mp4"),
    use_gemini=True,
    user_prompt="Analyze this e-commerce checkout flow"
)
```

## 🏗️ Architecture

### Core Components

```
video-use/
├── video_use/
│   ├── analysis/           # Analysis services
│   │   ├── services.py     # Video analysis & Gemini services
│   │   └── __init__.py     # Analysis module exports
│   ├── models.py          # Core data models and configurations
│   ├── services.py        # Main business logic services
│   ├── prompts.py         # LLM prompts for analysis
│   └── __init__.py        # Package exports
├── examples/              # Usage examples
│   ├── simple_example.py  # Basic usage demonstration
│   ├── frame-extraction/  # Frame extraction examples
│   └── sample_form_filling.mp4  # Demo video
└── tests/                 # Test suite
```

### Data Flow

1. **Video Input**: MP4, AVI, MOV, MKV, WebM format support
2. **Frame Extraction**: Intelligent sampling based on visual changes and FPS settings
3. **Analysis Processing**: Choice between AI-powered Gemini analysis or traditional frame processing
4. **Workflow Generation**: Convert analysis results into structured workflows
5. **Export**: Generate browser-use compatible prompt

## 🔌 Integration with Browser-Use

Video-Use generates structured workflows that can be integrated with [browser-use](https://github.com/browser-use/browser-use):

```python
from video_use import VideoUseService
from browser_use import Agent

# 1. Analyze video to get workflow description
service = VideoUseService()
result = await service.analyze_video_file(
    Path("login_demo.mp4"),
    use_gemini=True
)

# 2. Generate structured workflow
if result.success:
    workflow = await service.generate_structured_workflow_from_gemini(
        result.workflow_steps[0]['analysis_text'],
        start_url="https://example.com/login"
    )
    
    # 3. Use the workflow prompt with browser-use Agent
    agent = Agent()
    await agent.run(workflow.prompt)
    
    # The workflow will contain:
    # - Natural language description of actions
    # - Start URL for the automation
    # - Extracted parameters and values
    print(f"Workflow: {workflow.prompt}")
    print(f"Start URL: {workflow.start_url}")
    print(f"Parameters: {workflow.parameters}")
```

## 🤖 AI Models

### Primary AI Engine
- **Google Gemini 1.5 Pro**: Advanced multimodal AI for video understanding and action analysis
- **Frame processing**: Intelligent sampling and visual change detection using OpenCV
- **Natural language processing**: Converts video analysis into human-readable workflow descriptions

### Computer Vision
- **OpenCV**: Video processing, frame extraction, and visual change detection
- **Frame analysis**: Smart sampling based on visual differences and configured FPS


## 🧪 Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/your-org/video-use.git
cd video-use

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"

```

## 🚧 Roadmap

### High Priority
- **End-to-end testing integration**: Implement automated testing where Browser Use Agent executes the generated workflows to validate accuracy
- **Parameterized workflow execution**: Support dynamic values in workflows (e.g., CSV data input for batch form filling)
- **Workflow validation**: Add validation checks to ensure generated prompts produce expected results

### Medium Priority  
- **Enhanced error handling**: Better error messages and recovery strategies for failed video analysis
- **Performance optimization**: Optimize frame extraction and analysis for longer videos with image based models.


## 🙏 Acknowledgments

- [browser-use](https://github.com/browser-use/browser-use) - Browser automation framework
- [OpenCV](https://opencv.org/) - Computer vision library

---

**Made with ❤️ for the browser automation community**
