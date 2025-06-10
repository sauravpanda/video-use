# Video-Use Examples

This directory contains examples demonstrating how to use the video-use library for analyzing user interface interactions in videos.

## Available Examples

### `simple_example.py`
Basic demonstration of video analysis and workflow generation.

```bash
cd examples
python simple_example.py sample_form_filling.mp4
```

**Features shown:**
- Video loading and analysis
- Gemini AI analysis
- Structured workflow generation
- Token usage tracking

### `workflow_execution_example.py`
Complete pipeline demonstration including workflow execution with browser-use.

```bash
cd examples
export GOOGLE_API_KEY="your-gemini-api-key"
python workflow_execution_example.py sample_form_filling.mp4

# With options
python workflow_execution_example.py my_video.mp4 --mode individual --headless
```

**Features shown:**
- Complete pipeline: analyze → generate → execute
- Individual step control for advanced use cases
- Browser automation execution
- Error handling and result tracking
- Execution timeout and headless mode configuration
- Command line argument support for flexible video input

### `csv_batch_execution_example.py`
CSV-based batch processing for executing workflows with multiple data sets.

```bash
cd examples
export GOOGLE_API_KEY="your-gemini-api-key"
python csv_batch_execution_example.py sample_form_filling.mp4

# With custom CSV and options
python csv_batch_execution_example.py login_demo.mp4 user_data.csv --max-concurrent 3 --timeout 45
```

**Features shown:**
- Single video analysis for workflow template creation
- CSV data loading and processing
- Dynamic workflow customization with data placeholders
- Concurrent batch execution with rate limiting
- Comprehensive execution reporting and statistics
- Configurable parameters via command line arguments

### `frame-extraction/`
Advanced examples focusing on frame extraction and computer vision analysis.

```bash
cd examples/frame-extraction
python example_frame_extraction.py ../sample_form_filling.mp4 --mode frames
python example_frame_extraction.py ../sample_form_filling.mp4 --mode gemini
```

**Features shown:**
- Frame extraction with different sampling rates
- Visual change detection
- Computer vision analysis
- AI-powered action detection

## Sample Data

### `sample_form_filling.mp4`
A sample video showing form filling interactions. This video demonstrates:
- Navigating to a login page
- Filling out username and password fields
- Clicking submit button
- Form validation

Use this video to test the analysis capabilities and understand the expected input format.

### `sample_data.csv` (Auto-generated)
The CSV batch processing example automatically creates a sample CSV file with the following structure:
```csv
username,password,first_name,last_name,company
user1@example.com,password123,John,Doe,Acme Corp
user2@example.com,secret456,Jane,Smith,Tech Solutions
user3@example.com,mypass789,Bob,Johnson,StartupXYZ
```

This CSV demonstrates how to structure data for batch processing with dynamic workflow execution.

## Setup Requirements

### For Basic Examples
```bash
pip install opencv-python numpy
```

### For AI Analysis (Gemini)
```bash
pip install langchain-google-genai
export GOOGLE_API_KEY="your-gemini-api-key"
```

### For Workflow Execution
Ensure browser-use is properly installed (included in video-use dependencies) and that you have appropriate browser drivers.

### For Frame Extraction Examples
All dependencies are included in the main video-use package.

## Usage Tips

1. **Video Quality**: Use high-resolution videos (1080p+) for best results
2. **Clear Actions**: Ensure mouse movements and clicks are clearly visible
3. **Stable UI**: Avoid overlapping windows or UI animations during recording
4. **API Keys**: Set up your Gemini API key for AI-powered analysis
5. **CSV Format**: Use clear column names that match the actions in your video (e.g., username, password, email)
6. **Placeholders**: Use `{column_name}` format in your workflow prompts for dynamic data replacement

## Workflow Execution Features

### Complete Pipeline
The `workflow_execution_example.py` demonstrates the full pipeline:
1. **Video Analysis**: Analyze browser interaction video with Gemini AI
2. **Workflow Generation**: Create structured, executable workflow instructions
3. **Browser Execution**: Execute the workflow using browser-use agent
4. **Result Validation**: Track execution success and performance metrics

### CSV Batch Processing
The `csv_batch_execution_example.py` enables:
1. **Template Creation**: Analyze video once to create a reusable workflow template
2. **Data Loading**: Load multiple data sets from CSV files
3. **Dynamic Execution**: Execute the same workflow with different data for each row
4. **Concurrent Processing**: Process multiple workflows simultaneously with rate limiting
5. **Comprehensive Reporting**: Track success rates, execution times, and error details

## Troubleshooting

### Common Issues

**"Video file not found"**
- Ensure the video file path is correct
- Check that the video file is in a supported format (MP4, AVI, MOV, MKV, WebM)

**"API key not set"**
- Set the GOOGLE_API_KEY environment variable
- Verify your API key has access to Gemini models

**"Analysis failed"**
- Check video quality and format
- Ensure the video contains clear browser interactions
- Review the error logs for specific issues

**"Workflow execution failed"**
- Verify browser-use is properly installed
- Check that the target website is accessible
- Ensure the workflow instructions are clear and actionable
- Review browser console for JavaScript errors

**"CSV batch processing issues"**
- Verify CSV file format and encoding (UTF-8 recommended)
- Check that column names match expected placeholders in the workflow
- Review individual execution errors in the batch results

### Performance Optimization

**For Large CSV Files:**
- Adjust `max_concurrent` parameter to balance speed vs. resource usage
- Increase `timeout` for complex workflows
- Use `headless=True` for faster execution

**For Better Analysis:**
- Record videos at consistent speed
- Ensure good lighting and contrast
- Avoid rapid mouse movements or complex animations 