# Testing Guide for Video-Use

This guide provides step-by-step instructions for testing the video-use library.

## 🚀 Quick Start Testing

### 1. Test Without Video (Synthetic Data)
```bash
cd video-use
python examples/test_with_sample_data.py
```

**Expected Output:**
```
🧪 Video-Use Library Test Suite
========================================
This test runs without requiring a real video file.
It uses synthetic data to validate the processing pipeline.

🔍 Testing UI Detection
-------------------------
📊 Simulating UI detection on frame 0
   Found elements:
   - Input field (email) at (100, 150)
   - Input field (password) at (100, 200)
   - Button (Login) at (150, 250)

🧪 Testing Action Inference
------------------------------
✅ Inferred X actions:
   1. TYPE: Type email in input field
      Confidence: 0.88
      Timestamp: 1.0s
      Value: john@example.com

   2. TYPE: Type password in input field  
      Confidence: 0.85
      Timestamp: 2.0s
      Value: ••••••••

   3. CLICK: Click on Login
      Confidence: 0.95
      Timestamp: 3.0s

🚀 Testing Full Pipeline
==============================
📋 Generated Workflow Summary:
   - Total actions detected: 3
   - type: 2
   - click: 1
   - Average confidence: 0.89

🎯 Expected workflow:
   1. Type email in username field
   2. Type password in password field
   3. Click login button
   4. Navigate to dashboard

✅ Pipeline test completed!

🎉 All tests completed successfully!

📝 Next steps:
1. Try the CLI: `video-use demo`
2. Run with real video: `python examples/simple_example.py`
3. Record a browser interaction and analyze it
```

### 2. Test CLI Commands
```bash
# Show demo
video-use demo

# Show configuration
video-use config --show

# List supported formats
video-use info --help
```

## 🎥 Testing With Real Videos

### Step 1: Record a Test Video

**Recording Setup:**
- **Screen recorder**: OBS Studio, QuickTime (Mac), or Windows Game Bar
- **Resolution**: 1080p minimum
- **Frame rate**: 30fps or higher
- **Duration**: 15-60 seconds for testing

**Recording a Simple Login Flow:**
1. Open your browser
2. Navigate to a login page (e.g., Gmail, GitHub, any test site)
3. Start recording
4. Slowly and deliberately:
   - Click on username field
   - Type username
   - Click on password field  
   - Type password
   - Click login button
   - Wait for page load
5. Stop recording
6. Save as `test_login.mp4`

### Step 2: Test With Your Video
```bash
# Place your video in the examples directory
cp /path/to/your/test_login.mp4 video-use/examples/test_video.mp4

# Run the simple example
cd video-use
python examples/simple_example.py
```

**Expected Output:**
```
🎥 Video-Use Simple Example
==================================================
🔍 Analyzing video: test_video.mp4
This may take a moment...

✅ Analysis completed successfully!
📊 Results:
   - Analysis ID: abc123-def456-789
   - Processing time: 23.45 seconds
   - Confidence score: 0.84
   - Workflow steps found: 5

🔄 Workflow Steps:
   1. CLICK: Click on username field
      Confidence: 0.92
      Target: email

   2. TYPE: Type username
      Confidence: 0.88

   3. CLICK: Click on password field  
      Confidence: 0.90
      Target: password

   4. TYPE: Type password
      Confidence: 0.85

   5. CLICK: Click login button
      Confidence: 0.95
      Target: Login

🔄 Exporting to browser-use format...
✅ Browser-use workflow generated!
📝 Workflow name: Form Filling Workflow
📝 Description: Workflow with 5 steps including 2 click actions, 2 type actions, 1 navigate action. Duration: 12.3 seconds.
💾 Saved workflow to: exported_workflow.json

🤖 Sample browser-use integration:
```python
from browser_use import Agent
import json

# Load the exported workflow
with open('exported_workflow.json', 'r') as f:
    workflow = json.load(f)

# Execute with browser-use
agent = Agent()
await agent.execute_workflow(workflow)
```
```

### Step 3: Test CLI with Real Video
```bash
# Basic analysis
video-use analyze examples/test_video.mp4

# With output directory
video-use analyze examples/test_video.mp4 --output ./test_results

# Quick analysis
video-use analyze examples/test_video.mp4 --quick

# With context prompt
video-use analyze examples/test_video.mp4 --prompt "User logging into email account"

# Export the workflow
video-use list  # Get the analysis ID
video-use export <analysis-id> --format browser-use --output login_workflow.json
```

## 🧪 Advanced Testing Scenarios

### Test 1: Form Filling
**Record yourself:**
- Filling out a contact form
- Registration form
- Survey form

**Test Command:**
```bash
video-use analyze form_test.mp4 \
  --prompt "Fill out contact form with name, email, and message" \
  --fps 1.5 \
  --confidence 0.7 \
  --output ./form_results
```

### Test 2: E-commerce Flow
**Record yourself:**
- Adding product to cart
- Checkout process
- Payment form

**Test Command:**
```bash
video-use analyze shopping_test.mp4 \
  --prompt "Add product to cart and complete checkout" \
  --fps 2.0 \
  --output ./shopping_results
```

### Test 3: Navigation Flow
**Record yourself:**
- Browsing through website menu
- Clicking different pages
- Using search functionality

**Test Command:**
```bash
video-use analyze navigation_test.mp4 \
  --prompt "Navigate through website menu and search" \
  --quick \
  --output ./nav_results
```

## 🔧 Performance Testing

### Test Processing Speed
```bash
# Time the analysis
time video-use analyze test_video.mp4 --quick

# Test with different settings
time video-use analyze test_video.mp4 --fps 0.5  # Faster
time video-use analyze test_video.mp4 --fps 2.0  # Slower but more accurate
```

### Test Memory Usage
```bash
# Monitor memory during analysis
/usr/bin/time -v video-use analyze test_video.mp4 --verbose
```

### Test Different Video Formats
```bash
# Test various formats
video-use info test.mp4
video-use info test.avi  
video-use info test.mov
video-use info test.mkv
```

## 🚨 Error Testing

### Test Invalid Videos
```bash
# Test with non-video file
video-use analyze README.md  # Should fail gracefully

# Test with corrupted video
video-use analyze corrupted.mp4  # Should handle errors

# Test with very large video
video-use analyze huge_video.mp4 --quick  # Should limit processing
```

### Test Edge Cases
```bash
# Very short video
video-use analyze 3_second_video.mp4

# Very long video
video-use analyze 10_minute_video.mp4 --quick

# Low resolution video
video-use analyze 480p_video.mp4 --confidence 0.6
```

## 📊 Validation Testing

### Manual Validation
1. **Record a known sequence** (e.g., login with known steps)
2. **Analyze the video**
3. **Compare results** with actual steps:
   - Are all actions detected?
   - Are action types correct?
   - Are timestamps reasonable?
   - Is confidence scoring meaningful?

### Expected Accuracy
For well-recorded videos:
- **Click detection**: 85-95% accuracy
- **Text input detection**: 80-90% accuracy  
- **Navigation detection**: 75-85% accuracy
- **Overall workflow**: 80-90% usefulness

### Validation Checklist
- [ ] All major actions detected
- [ ] Action sequence is logical
- [ ] Timestamps are reasonable
- [ ] UI elements properly identified
- [ ] Exported workflow is executable
- [ ] Performance is acceptable

## 🐛 Troubleshooting Common Issues

### Issue: No actions detected
**Solutions:**
```bash
# Lower confidence threshold
video-use analyze video.mp4 --confidence 0.5

# Increase frame sampling  
video-use analyze video.mp4 --fps 2.0

# Enable verbose logging
video-use analyze video.mp4 --verbose
```

### Issue: Too many false positives
**Solutions:**
```bash
# Increase confidence threshold
video-use analyze video.mp4 --confidence 0.9

# Use quick mode
video-use analyze video.mp4 --quick

# Provide context
video-use analyze video.mp4 --prompt "Specific task description"
```

### Issue: Poor performance
**Solutions:**
```bash
# Use quick mode
video-use analyze video.mp4 --quick

# Reduce frame rate
video-use analyze video.mp4 --fps 0.5

# Limit max frames
# (Edit config in code to set max_frames=100)
```

### Issue: Video format not supported
**Solutions:**
```bash
# Check video info first
video-use info video.ext

# Convert to MP4 using ffmpeg
ffmpeg -i input.avi output.mp4

# Use supported formats: MP4, AVI, MOV, MKV, WebM
```

## 📈 Performance Benchmarks

### Expected Processing Times
| Video Length | Frames | Quick Mode | Full Mode | Actions Expected |
|-------------|---------|------------|-----------|------------------|
| 15 seconds  | 15      | 5-10s      | 15-25s    | 2-5             |
| 30 seconds  | 30      | 8-15s      | 25-40s    | 5-10            |
| 1 minute    | 60      | 15-25s     | 45-70s    | 10-20           |
| 2 minutes   | 120     | 25-40s     | 80-120s   | 20-40           |

*Results on modern laptop (Intel i7, 16GB RAM)*

## 🎯 Success Criteria

A successful test should demonstrate:

1. **Functional Pipeline**: All components work together
2. **Reasonable Accuracy**: Detects 70%+ of obvious actions
3. **Good Performance**: Processes 1 minute video in under 2 minutes
4. **Browser-Use Integration**: Exports valid workflow format
5. **Error Handling**: Gracefully handles invalid inputs
6. **User Experience**: Clear CLI output and helpful error messages

## 📝 Reporting Issues

When reporting issues, include:

1. **Video details**: Resolution, duration, format, content type
2. **Command used**: Full command line with all parameters
3. **Expected vs actual results**: What you expected vs what happened
4. **Error logs**: Full error output with `--verbose` flag
5. **System info**: OS, Python version, available memory

```bash
# Generate comprehensive debug info
video-use analyze problem_video.mp4 --verbose 2>&1 | tee debug.log
video-use info problem_video.mp4 >> debug.log
python --version >> debug.log
```

This testing guide should help you thoroughly validate the video-use library and identify any issues that need to be addressed. 