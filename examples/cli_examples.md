# CLI Examples for Video-Use

This file contains various examples of how to use the video-use command line interface.

## 📋 Basic Usage

### Analyze a video file
```bash
video-use analyze my_recording.mp4
```

### Analyze with output directory
```bash
video-use analyze my_recording.mp4 --output ./analysis_results
```

### Quick analysis (keyframes only)
```bash
video-use analyze my_recording.mp4 --quick
```

### Verbose output
```bash
video-use analyze my_recording.mp4 --verbose
```

## ⚙️ Configuration Examples

### Custom frame extraction rate
```bash
video-use analyze recording.mp4 --fps 2.0
```

### Lower confidence threshold (more detections)
```bash
video-use analyze recording.mp4 --confidence 0.6
```

### Combined settings
```bash
video-use analyze recording.mp4 --fps 1.5 --confidence 0.7 --output ./results --verbose
```

## 🎯 Context-Aware Analysis

### With user prompt for better context
```bash
video-use analyze form_filling.mp4 --prompt "User filling out a registration form"
```

### E-commerce workflow
```bash
video-use analyze shopping.mp4 --prompt "Customer adding items to cart and checking out"
```

### Login workflow
```bash
video-use analyze login.mp4 --prompt "User logging into admin dashboard"
```

## 📤 Export Examples

### Export to browser-use format
```bash
video-use export abc123 --format browser-use --output workflow.json
```

### View workflow in terminal
```bash
video-use export abc123 --format browser-use
```

### Export raw analysis as JSON
```bash
video-use export abc123 --format json --output full_analysis.json
```

## 🔧 Utility Commands

### Show video information
```bash
video-use info my_video.mp4
```

### List all cached analyses
```bash
video-use list
```

### Show current configuration
```bash
video-use config --show
```

### Clean cached analyses
```bash
video-use clean --yes
```

### Show demo and help
```bash
video-use demo
```

## 🎬 Real-World Examples

### Example 1: Form Submission
```bash
# Record yourself filling out a contact form
video-use analyze contact_form.mp4 \
  --prompt "Fill out contact form with name, email, and message" \
  --output ./contact_form_analysis \
  --fps 1.0

# Export for automation
video-use export <analysis-id> \
  --format browser-use \
  --output contact_form_workflow.json
```

### Example 2: E-commerce Shopping
```bash
# Record adding items to cart and checkout
video-use analyze shopping_flow.mp4 \
  --prompt "Add product to cart and complete checkout process" \
  --confidence 0.8 \
  --output ./shopping_analysis

# Export workflow
video-use export <analysis-id> \
  --format browser-use \
  --output shopping_workflow.json
```

### Example 3: Admin Dashboard Navigation
```bash
# Record navigating through admin interface
video-use analyze admin_nav.mp4 \
  --prompt "Navigate through admin dashboard and update settings" \
  --quick \
  --output ./admin_analysis

# View results
video-use list
video-use export <analysis-id> --format browser-use
```

## 📊 Performance Optimization

### Fast analysis for long videos
```bash
video-use analyze long_video.mp4 \
  --quick \
  --fps 0.5 \
  --confidence 0.8
```

### High accuracy for short videos
```bash
video-use analyze short_video.mp4 \
  --fps 2.0 \
  --confidence 0.9 \
  --verbose
```

### Batch processing multiple videos
```bash
for video in *.mp4; do
  echo "Processing $video..."
  video-use analyze "$video" \
    --output "./results/$(basename "$video" .mp4)" \
    --quick
done
```

## 🚨 Error Handling

### Check video compatibility first
```bash
video-use info suspicious_video.avi
```

### Analyze with error recovery
```bash
video-use analyze problematic_video.mp4 \
  --confidence 0.6 \
  --fps 0.5 \
  --verbose 2>&1 | tee analysis.log
```

## 💡 Pro Tips

### 1. Start with quick analysis
```bash
# Get a fast overview
video-use analyze video.mp4 --quick

# If results look good, do full analysis
video-use analyze video.mp4 --fps 2.0 --output ./detailed_results
```

### 2. Use prompts for better results
```bash
# Generic analysis
video-use analyze video.mp4

# Context-aware analysis (better results)
video-use analyze video.mp4 --prompt "User booking a flight ticket"
```

### 3. Adjust settings based on video type
```bash
# For UI-heavy videos (lots of buttons/forms)
video-use analyze ui_heavy.mp4 --confidence 0.7 --fps 1.5

# For simple navigation videos
video-use analyze simple_nav.mp4 --quick --confidence 0.8
```

### 4. Save and reuse configurations
```bash
# Create alias for common settings
alias video-analyze-form='video-use analyze --fps 1.5 --confidence 0.7 --verbose'

# Use it
video-analyze-form my_form_video.mp4 --output ./results
```

## 🔗 Integration with Browser-Use

### Complete workflow
```bash
# 1. Analyze video
video-use analyze checkout.mp4 \
  --prompt "Complete e-commerce checkout process" \
  --output ./checkout_analysis

# 2. Export workflow  
video-use export <analysis-id> \
  --format browser-use \
  --output checkout_workflow.json

# 3. Use with browser-use (in Python)
# python -c "
# import json
# from browser_use import Agent
# 
# with open('checkout_workflow.json') as f:
#     workflow = json.load(f)
# 
# agent = Agent()
# await agent.execute_workflow(workflow)
# "
```

## 📈 Monitoring and Analysis

### Get analysis statistics
```bash
# List all analyses with details
video-use list

# Get specific analysis info
video-use export <analysis-id> --format json
```

### Performance monitoring
```bash
# Time the analysis
time video-use analyze video.mp4 --output ./timed_results

# Monitor with verbose output
video-use analyze video.mp4 --verbose 2>&1 | grep -E "(Processing|Completed|Error)"
``` 