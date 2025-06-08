# Testing with Real Videos

This guide walks you through testing the video-use library with actual video recordings of browser interactions.

## Quick Start

### 1. Record a Video

First, record yourself performing a browser interaction:

**For macOS:**
```bash
# Built-in screen recording (Cmd+Shift+5)
# Or use QuickTime Player > File > New Screen Recording
```

**For Windows:**
```bash
# Windows 10/11: Windows Key + G (Game Bar)
# Or use built-in Screen Sketch tool
```

**For Linux:**
```bash
# Using ffmpeg
ffmpeg -f x11grab -s 1920x1080 -i :0.0 output.mp4

# Using OBS Studio (cross-platform)
# Free and powerful screen recording
```

### 2. Test Your Video

```bash
cd video-use/examples
python test_real_video.py path/to/your/recording.mp4
```

## Recording Best Practices

### What Makes a Good Test Video

1. **Clear Actions** - Make deliberate, distinct movements
2. **Visible Elements** - Ensure UI elements are clearly visible
3. **Reasonable Pace** - Don't move too fast between actions
4. **Good Resolution** - Use at least 1080p for better OCR results
5. **Stable Frame Rate** - 30fps is ideal for analysis

### Recommended Test Scenarios

#### 1. Simple Login Flow (30-60 seconds)
```
1. Navigate to login page
2. Click username field
3. Type username
4. Click password field  
5. Type password
6. Click login button
```

#### 2. Form Filling (1-2 minutes)
```
1. Navigate to form page
2. Fill out multiple fields
3. Select dropdown options
4. Check/uncheck boxes
5. Submit form
```

#### 3. E-commerce Flow (2-3 minutes)
```
1. Search for product
2. Click on product
3. Add to cart
4. Go to checkout
5. Fill out form fields
```

## Testing Commands

### Basic Test
```bash
python test_real_video.py my_recording.mp4
```

### With Context Prompt
```bash
python test_real_video.py login_demo.mp4 --prompt "User logging into Gmail"
```

### Quick Analysis (Faster)
```bash
python test_real_video.py long_video.mp4 --quick
```

### Verbose Output (Debug)
```bash
python test_real_video.py my_video.mp4 --verbose
```

### Combined Options
```bash
python test_real_video.py checkout_flow.mp4 \
  --prompt "User completing online purchase" \
  --quick \
  --verbose
```

## Expected Output

### Success Example
```
🎥 Video-Use Real Video Test
==================================================
📁 Video file: login_demo.mp4
💭 User prompt: User logging into website
⚡ Quick mode: No

📊 Video file size: 15.2 MB
⚙️  Using full analysis configuration

🔍 Starting analysis...
⏳ This may take a moment depending on video length...

✅ Analysis completed successfully!
📊 Results Summary:
   🆔 Analysis ID: vid_20231201_143052
   ⏱️  Processing time: 28.45 seconds
   🎯 Confidence score: 0.87
   📋 Workflow steps: 3

🔄 Detected Workflow Steps:
------------------------------------------------------------
 1. TYPE: Enter username in login field
     ⏱️  Time: 5.2s | 🎯 Confidence: 0.92
     🎯 Target: 'Username'
     📍 Position: (320, 180) size: 200x30
     💬 Value: 'user@example.com'

 2. TYPE: Enter password in password field
     ⏱️  Time: 12.8s | 🎯 Confidence: 0.88
     🎯 Target: 'Password'
     📍 Position: (320, 230) size: 200x30
     💬 Value: '********'

 3. CLICK: Click login button
     ⏱️  Time: 18.5s | 🎯 Confidence: 0.91
     🎯 Target: 'Login'
     📍 Position: (420, 280) size: 80x35

🔄 Exporting to browser-use format...
✅ Browser-use workflow generated!
📝 Workflow Details:
   Name: Video Analysis Workflow
   Description: User logging into website
   Steps: 3

💾 Saved workflow to: /path/to/login_demo_workflow.json

🤖 Sample Browser-Use Commands:
----------------------------------------
1. TYPE
   Selector: input[placeholder*="username" i], input[name*="username" i]
   Text: user@example.com

2. TYPE
   Selector: input[type="password"]
   Text: ********

3. CLICK
   Selector: button:contains("Login"), input[type="submit"]

🔗 Integration Example:
```python
from browser_use import Agent
import json

# Load the exported workflow
with open('login_demo_workflow.json', 'r') as f:
    workflow = json.load(f)

# Execute with browser-use
agent = Agent()
await agent.execute_workflow(workflow)
```

🎉 Test completed successfully!
📁 Results saved in current directory
```

## Troubleshooting

### Common Issues

#### Video Not Found
```
❌ Error: Video file not found: /path/to/video.mp4
📝 Make sure the video file exists and try again.
```
**Solution:** Check the file path and ensure the video exists.

#### Unsupported Format
```
❌ Error: Unsupported video format
📋 Supported formats: ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
```
**Solution:** Convert your video to a supported format:
```bash
# Using ffmpeg
ffmpeg -i input.format output.mp4
```

#### Low Confidence Results
```
🎯 Confidence score: 0.45
```
**Solutions:**
1. Record at higher resolution
2. Move mouse more slowly
3. Ensure good lighting/contrast
4. Use `--prompt` to provide context
5. Record simpler interactions

#### Processing Takes Too Long
**Solutions:**
1. Use `--quick` flag for faster processing
2. Record shorter videos (under 2 minutes)
3. Use lower resolution recordings
4. Trim video to essential parts

### Performance Tips

#### Optimal Video Specs
- **Resolution:** 1920x1080 (1080p)
- **Frame Rate:** 30fps
- **Duration:** 30 seconds - 2 minutes
- **Format:** MP4 (H.264)
- **File Size:** Under 50MB

#### Processing Time Estimates
- **30-second video:** 15-30 seconds
- **1-minute video:** 30-60 seconds  
- **2-minute video:** 60-120 seconds
- **Quick mode:** ~50% faster

## Advanced Testing

### Custom Configuration
Create a custom config file:

```python
# custom_config.py
from video_use import VideoAnalysisConfig

config = VideoAnalysisConfig(
    frame_extraction_fps=2.0,      # Higher frame rate
    ui_detection_confidence=0.8,   # Higher confidence
    enable_ocr=True,
    max_frames=300,                # More frames
    parallel_processing=True,
    ocr_languages=['eng', 'spa'],  # Multiple languages
)
```

### Batch Testing
Test multiple videos:

```bash
for video in *.mp4; do
    echo "Testing $video..."
    python test_real_video.py "$video" --quick
done
```

### Integration Testing
Test the exported workflow with browser-use:

```python
import json
from browser_use import Agent

# Load exported workflow
with open('my_video_workflow.json', 'r') as f:
    workflow = json.load(f)

# Test with browser-use
agent = Agent()
result = await agent.execute_workflow(workflow)
print(f"Workflow executed: {result.success}")
```

## Sample Videos for Testing

### Create Test Scenarios

#### 1. Login Test
1. Go to any login page (Gmail, GitHub, etc.)
2. Enter credentials (use test account)
3. Click login
4. Record 30-60 seconds

#### 2. Search Test  
1. Go to search page (Google, Amazon, etc.)
2. Type search query
3. Click search or press Enter
4. Record 20-30 seconds

#### 3. Form Test
1. Go to contact form or signup page
2. Fill multiple fields
3. Select options from dropdowns
4. Submit form
5. Record 1-2 minutes

### Recording Tips
- Use consistent mouse movements
- Don't move too fast between elements
- Make sure text is readable
- Include context in the recording
- Test one clear workflow per video

## Next Steps

After successful testing:

1. **Export Workflows** - Save the generated JSON files
2. **Integrate with Browser-Use** - Use in your automation scripts
3. **Optimize Settings** - Adjust config for your use cases
4. **Scale Testing** - Test with various websites and scenarios
5. **Production Use** - Deploy in your automation pipeline

Ready to test with real videos? Start with a simple login flow and work up to more complex interactions! 