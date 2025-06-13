# Video-Use

**Convert browser interaction videos into automated workflows using AI and computer vision.**

Video-Use analyzes screen recordings of browser interactions and automatically generates executable workflows compatible with [browser-use](https://github.com/browser-use/browser-use) for automation.

## 🎯 Core Features

1. **Video Analysis**: Extract and analyze browser interaction videos using AI (Gemini)
2. **Workflow Generation**: Convert analyzed actions into structured workflows
3. **Workflow Execution**: Execute generated workflows using browser-use automation
4. **Batch Processing**: Support for executing workflows with multiple data sets via CSV

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
    
    # Analyze video with Gemini AI
    result = await service.analyze_video_file(
        Path("recording.mp4"),
        use_gemini=True
    )
    
    if result.success:
        # Generate and execute workflow
        workflow = await service.generate_structured_workflow_from_gemini(
            result.workflow_steps[0]['analysis_text'],
            start_url="https://example.com"
        )
        execution_result = await service.execute_workflow(workflow)
        
        if execution_result.success:
            print(f"Workflow executed successfully!")
        else:
            print(f"Execution failed: {execution_result.error_message}")

asyncio.run(main())
```

## 📋 Key Components

### Video Analysis Service
- Frame extraction from browser interaction videos
- AI-powered analysis using Google Gemini
- Structured workflow generation
- Support for MP4, AVI, MOV, MKV, WebM formats

### Workflow Execution Service
- Browser automation using browser-use
- Configurable execution parameters (timeout, headless mode)
- Shared browser session support
- Execution status tracking and management

### Batch Processing
- CSV-based data input
- Dynamic workflow customization
- Concurrent execution support
- Comprehensive execution reporting

## 🛠️ Configuration

### Environment Variables

```bash
# Required for Gemini AI analysis
export GOOGLE_API_KEY="your-gemini-api-key"

# Optional: specify API endpoint
export GOOGLE_API_BASE="https://generativelanguage.googleapis.com"
```

### Analysis Configuration

```python
from video_use import VideoAnalysisConfig

config = VideoAnalysisConfig(
    frame_extraction_fps=1.0,    # Extract 1 frame per second
    max_frames=20,              # Maximum frames to process
)
```

### Execution Configuration

```python
# Workflow execution options
execution_result = await service.execute_workflow(
    workflow,
    headless=False,            # Run browser in visible mode
    timeout=30,               # Execution timeout in seconds
    use_shared_session=True   # Use shared browser session
)
```

## 📊 Example Output

### AI Analysis Results
```python
workflow = StructuredWorkflowOutput(
    prompt="Navigate to login page, fill out username and password, then submit the form",
    start_url="https://example.com/login",
    parameters={
        "username": "user@example.com",
        "password": "[HIDDEN]",
        "login_button_text": "Login"
    }
)
```

### Execution Results
```python
execution_response = WorkflowExecutionResponse(
    success=True,
    execution_id="550e8400-e29b-41d4-a716-446655440000",
    results=[{"workflow_result": "Workflow completed successfully"}],
    execution_time=12.5
)
```

## 🎥 Recording Guidelines

For optimal analysis results:

1. **Video Quality**
   - Use 1080p or higher resolution
   - Record at 30fps or higher
   - Ensure good lighting and contrast

2. **Browser Setup**
   - Use full screen or consistent window size
   - Avoid overlapping windows
   - Keep UI elements clearly visible

3. **Interaction Best Practices**
   - Move mouse deliberately and smoothly
   - Click precisely on target elements
   - Pause briefly between actions
   - Type at normal speed

## 🔧 Examples

See the [examples directory](examples/README.md) for detailed usage examples:

- Basic video analysis and workflow generation
- Complete workflow execution pipeline
- CSV-based batch processing
- Frame extraction and analysis

## 📝 Requirements

- Python 3.8+
- OpenAI API key (for GPT-4)
- Google API key (for Gemini)
- Modern web browser (Chrome recommended)

## 🔍 Troubleshooting

### Common Issues

1. **Video Analysis Fails**
   - Verify video format and quality
   - Check API key configuration
   - Ensure clear browser interactions

2. **Workflow Execution Fails**
   - Verify browser-use installation
   - Check website accessibility
   - Review browser console for errors

3. **Batch Processing Issues**
   - Verify CSV format and encoding
   - Check column names match workflow parameters
   - Review individual execution errors

For more detailed troubleshooting, see the [examples README](examples/README.md).

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

Video-Use provides seamless integration with [browser-use](https://github.com/browser-use/browser-use) through multiple approaches:

### Direct Execution (Recommended)
```python
from video_use import VideoUseService

# Complete pipeline with automatic execution
service = VideoUseService()
results = await service.analyze_and_execute_workflow(
    Path("login_demo.mp4"),
    start_url="https://example.com/login",
    use_gemini=True,
    headless=True,
    timeout=60
)

if results["success"]:
    print("Workflow executed successfully!")
    print(f"Analysis: {results['analysis']}")
    print(f"Execution time: {results['execution'].execution_time}s")
else:
    print(f"Pipeline failed: {results.get('error', 'Unknown error')}")
```

### Manual Integration
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
    
    # 3. Execute using video-use's built-in execution service
    execution_result = await service.execute_workflow(workflow)
    
    # OR manually use browser-use Agent
    agent = Agent()
    await agent.run(workflow.prompt)
    
    # The workflow contains:
    # - Natural language description of actions
    # - Start URL for the automation
    # - Extracted parameters and values
    print(f"Workflow: {workflow.prompt}")
    print(f"Start URL: {workflow.start_url}")
    print(f"Parameters: {workflow.parameters}")
```

### CSV Batch Processing
```python
from video_use import VideoUseService
from examples.csv_batch_execution_example import CSVBatchProcessor

# Batch process multiple data sets with same workflow
service = VideoUseService()
processor = CSVBatchProcessor(service)

# Analyze video once to create template
await processor.analyze_video_for_template(
    Path("form_filling_demo.mp4"),
    template_start_url="https://example.com/form"
)

# Load CSV data and execute batch
csv_data = processor.load_csv_data(Path("user_data.csv"))
results = await processor.execute_batch(
    csv_data,
    headless=True,
    max_concurrent=3
)

print(f"Processed {len(results)} workflows")
successful = sum(1 for r in results if r['success'])
print(f"Success rate: {successful}/{len(results)}")
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
- ✅ **End-to-end testing integration**: Implement automated testing where Browser Use Agent executes the generated workflows to validate accuracy
- ✅ **Parameterized workflow execution**: Support dynamic values in workflows (e.g., CSV data input for batch form filling)
- **Workflow validation**: Add validation checks to ensure generated prompts produce expected results

### Medium Priority  
- **Enhanced error handling**: Better error messages and recovery strategies for failed video analysis
- **Performance optimization**: Optimize frame extraction and analysis for longer videos with image based models.


## 🙏 Acknowledgments

- [browser-use](https://github.com/browser-use/browser-use) - Browser automation framework that powers our workflow execution
- [Google Gemini](https://deepmind.google/technologies/gemini/) - Advanced AI model that enables intelligent video analysis
- [OpenCV](https://opencv.org/) - Computer vision library

---

**Made with ❤️ for the browser automation community**
