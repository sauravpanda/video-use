"""
Simple example demonstrating video-use library usage.

This example shows how to:
1. Analyze a video file
2. Extract workflow steps  
3. Export to browser-use format
4. Display results
"""

import asyncio
import logging
from pathlib import Path
from video_use import VideoService, VideoAnalysisConfig

# Enable logging to see what's happening
logging.basicConfig(level=logging.INFO)

async def main():
    """Main example function."""
    
    print("🎥 Video-Use Simple Example")
    print("=" * 50)

    file_path = Path("sample_form_filling.mp4")
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    # Configuration for faster testing (you can adjust these)
    config = VideoAnalysisConfig(
        frame_extraction_fps=1.0,      # Extract 1 frame per second
        ui_detection_confidence=0.7,   # Lower confidence for more detections
        enable_ocr=True,               # Enable text detection
        max_frames=50,                 # Limit frames for faster testing
        parallel_processing=True       # Enable parallel processing
    )
    
    # Initialize the video service
    service = VideoService(config)
    
    # Example video path (you'll need to provide your own video)
    video_path = Path(file_path)
    
    # Check if video exists
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        print("\n📝 To test this example:")
        print("1. Record a short browser interaction video (10-30 seconds)")
        print(f"2. Save it as '{file_path}' in this directory")
        print("3. Run this script again")
        print("\n💡 Tip: Record yourself:")
        print("- Opening a website")
        print("- Clicking a button")
        print("- Filling out a form")
        print("- Navigating between pages")
        return
    
    try:
        print(f"🔍 Analyzing video: {video_path}")
        print("This may take a moment...")
        
        # Analyze the video
        result = await service.analyze_video_file(
            video_path,
            user_prompt="Analyze this browser interaction for automation"
        )
        
        if result.success:
            print(f"\n✅ Analysis completed successfully!")
            print(f"📊 Results:")
            print(f"   - Analysis ID: {result.analysis_id}")
            print(f"   - Processing time: {result.processing_time:.2f} seconds")
            print(f"   - Confidence score: {result.confidence_score:.2f}")
            print(f"   - Workflow steps found: {len(result.workflow_steps)}")
            
            # Display workflow steps
            if result.workflow_steps:
                print(f"\n🔄 Workflow Steps:")
                for i, step in enumerate(result.workflow_steps, 1):
                    print(f"   {i}. {step['action_type'].upper()}: {step['description']}")
                    if 'confidence' in step:
                        print(f"      Confidence: {step['confidence']:.2f}")
                    if 'target_element' in step and step['target_element']:
                        element = step['target_element']
                        if 'text' in element and element['text']:
                            print(f"      Target: {element['text']}")
                    print()
            
            # Export to browser-use format
            print("🔄 Exporting to browser-use format...")
            workflow = await service.export_workflow_to_browser_use(result.analysis_id)
            
            if workflow:
                print("✅ Browser-use workflow generated!")
                print(f"📝 Workflow name: {workflow['name']}")
                print(f"📝 Description: {workflow['description']}")
                
                # Save workflow to file
                import json
                output_file = Path("exported_workflow.json")
                with open(output_file, 'w') as f:
                    json.dump(workflow, f, indent=2)
                print(f"💾 Saved workflow to: {output_file}")
                
                # Display sample browser-use code
                print(f"\n🤖 Sample browser-use integration:")
                print("```python")
                print("from browser_use import Agent")
                print("import json")
                print("")
                print("# Load the exported workflow")
                print(f"with open('{output_file}', 'r') as f:")
                print("    workflow = json.load(f)")
                print("")
                print("# Execute with browser-use")
                print("agent = Agent()")
                print("await agent.execute_workflow(workflow)")
                print("```")
            
            else:
                print("❌ Failed to export workflow")
        
        else:
            print(f"❌ Analysis failed: {result.error_message}")
    
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("This might be due to missing dependencies or an unsupported video format")

if __name__ == "__main__":
    asyncio.run(main()) 