# Video-Use Examples

This directory contains examples demonstrating how to use the video-use library for analyzing user interface interactions in videos.

## Available Examples

### 1. Basic Example (`basic_example.py`)
A comprehensive example showing all main features of the library.

**Features:**
- Frame extraction from videos
- AI-powered analysis using Gemini
- VideoService unified interface
- Multiple analysis modes

**Usage:**
```bash
# Frame extraction mode
python basic_example.py sample_form_filling.mp4 --mode frames

# AI analysis mode (requires GOOGLE_API_KEY)
python basic_example.py sample_form_filling.mp4 --mode ai

# Traditional service analysis
python basic_example.py sample_form_filling.mp4 --mode service

# AI service analysis
python basic_example.py sample_form_filling.mp4 --mode service-ai
```

### 2. Frame Extraction Example (`frame-extraction/example_frame_extraction.py`)
Focused example for frame extraction and analysis.

**Usage:**
```bash
cd frame-extraction
python example_frame_extraction.py ../sample_form_filling.mp4 --mode frames
python example_frame_extraction.py ../sample_form_filling.mp4 --mode gemini
```

## Prerequisites

### For Frame Analysis
```bash
pip install opencv-python numpy
```

### For AI Analysis (Gemini)
```bash
pip install langchain-google-genai
export GOOGLE_API_KEY="your-gemini-api-key"
```

## Sample Video

The `sample_form_filling.mp4` file is included as a demonstration video showing a user filling out a web form. You can use your own videos by:

1. Recording browser interactions (10-30 seconds recommended)
2. Saving as MP4, AVI, MOV, MKV, or WebM format
3. Replacing the file path in the examples

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install video-use
   ```

2. **Run basic frame analysis:**
   ```bash
   python basic_example.py sample_form_filling.mp4
   ```

3. **Run AI analysis (with API key):**
   ```bash
   export GOOGLE_API_KEY="your-key"
   python basic_example.py sample_form_filling.mp4 --mode ai
   ```

## Example Outputs

### Frame Analysis
- Lists extracted frames with timestamps
- Shows frame numbers and timing information
- Useful for understanding video structure

### AI Analysis
- Provides step-by-step user action descriptions
- Identifies UI elements and interactions
- Generates professional UX analysis reports

### Service Analysis
- Returns structured workflow data
- Can export to browser-use compatible format
- Provides confidence scores and metadata

## Tips

- **Video Quality:** Higher quality videos provide better analysis results
- **Video Length:** Shorter videos (10-60 seconds) process faster
- **Recording:** Focus on clear UI interactions for best results
- **Format:** MP4 format generally works best 