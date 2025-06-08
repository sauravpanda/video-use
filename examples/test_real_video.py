"""
Test video-use library with a real video file.

Usage:
    python test_real_video.py path/to/your/video.mp4
    python test_real_video.py path/to/your/video.mp4 --prompt "User logging into website"
    python test_real_video.py path/to/your/video.mp4 --quick
"""

import asyncio
import sys
import argparse
import logging
import json
from pathlib import Path
from video_use import VideoService, VideoAnalysisConfig


def setup_logging(verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def print_header(video_path: Path, prompt: str = None, quick: bool = False):
    """Print analysis header information."""
    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    
    print("🎥 Video-Use Real Video Test")
    print("=" * 50)
    print(f"📁 Video file: {video_path}")
    print(f"📊 Video file size: {file_size_mb:.1f} MB")
    if prompt:
        print(f"💭 User prompt: {prompt}")
    print(f"⚡ Quick mode: {'Yes' if quick else 'No'}")
    print()


def create_config(quick: bool = False) -> VideoAnalysisConfig:
    """Create analysis configuration based on mode."""
    if quick:
        print("⚙️  Using quick analysis configuration")
        return VideoAnalysisConfig(
            frame_extraction_fps=0.5, ui_detection_confidence=0.6,
            enable_ocr=True, max_frames=50, parallel_processing=True
        )
    else:
        print("⚙️  Using full analysis configuration")
        return VideoAnalysisConfig(
            frame_extraction_fps=1.0, ui_detection_confidence=0.7,
            enable_ocr=True, max_frames=200, parallel_processing=True
        )


def print_results_summary(result):
    """Print analysis results summary."""
    print(f"\n✅ Analysis completed successfully!")
    print(f"📊 Results Summary:")
    print(f"   🆔 Analysis ID: {result.analysis_id}")
    print(f"   ⏱️  Processing time: {result.processing_time:.2f} seconds")
    print(f"   🎯 Confidence score: {result.confidence_score:.2f}")
    print(f"   📋 Workflow steps: {len(result.workflow_steps)}")


def print_workflow_steps(workflow_steps):
    """Print detailed workflow steps."""
    if not workflow_steps:
        return
        
    print(f"\n🔄 Detected Workflow Steps:")
    print("-" * 60)
    
    for i, step in enumerate(workflow_steps, 1):
        print(f"{i:2d}. {step['action_type'].upper()}: {step['description']}")
        print(f"     ⏱️  Time: {step['timestamp']:.1f}s | 🎯 Confidence: {step['confidence']:.2f}")
        
        # Show target element and value if available
        if step.get('target_element', {}).get('text'):
            print(f"     🎯 Target: '{step['target_element']['text']}'")
        if step.get('target_element', {}).get('bbox'):
            x, y, w, h = step['target_element']['bbox']
            print(f"     📍 Position: ({x}, {y}) size: {w}x{h}")
        if step.get('value'):
            print(f"     💬 Value: '{step['value']}'")
        print()


def save_and_show_workflow(workflow, video_path: Path):
    """Save workflow to file and show integration example."""
    if not workflow:
        print("❌ Failed to export workflow")
        return False
    
    print("✅ Browser-use workflow generated!")
    print(f"📝 Workflow Details:")
    print(f"   Name: {workflow['name']}")
    print(f"   Description: {workflow['description']}")
    print(f"   Steps: {len(workflow['steps'])}")
    
    # Save workflow to file
    output_file = video_path.stem + "_workflow.json"
    output_path = Path(output_file)
    
    with open(output_path, 'w') as f:
        json.dump(workflow, f, indent=2)
    print(f"💾 Saved workflow to: {output_path.absolute()}")
    
    # Show sample browser-use commands
    print(f"\n🤖 Sample Browser-Use Commands:")
    print("-" * 40)
    for i, step in enumerate(workflow['steps'][:3], 1):
        cmd = step['command']
        print(f"{i}. {cmd['type'].upper()}")
        for key in ['selector', 'text', 'coordinates']:
            if key in cmd:
                print(f"   {key.title()}: {cmd[key]}")
        print()
    
    if len(workflow['steps']) > 3:
        print(f"   ... and {len(workflow['steps']) - 3} more steps")
    
    # Show integration example
    print(f"\n🔗 Integration Example:")
    print("```python")
    print("from browser_use import Agent")
    print("import json")
    print("")
    print(f"with open('{output_file}', 'r') as f:")
    print("    workflow = json.load(f)")
    print("agent = Agent()")
    print("await agent.execute_workflow(workflow)")
    print("```")
    
    return True


def print_error_tips(verbose: bool = False):
    """Print troubleshooting tips."""
    if verbose:
        import traceback
        traceback.print_exc()
    print("\n💡 Troubleshooting tips:")
    print("1. Make sure the video file is not corrupted")
    print("2. Try with --quick flag for faster processing")
    print("3. Use --verbose flag to see detailed error info")
    print("4. Check that all dependencies are installed")


async def analyze_real_video(video_path: Path, prompt: str = None, quick: bool = False, verbose: bool = False):
    """Analyze a real video file."""
    
    print_header(video_path, prompt, quick)
    
    # Check if video exists
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        print("\n📝 Make sure the video file exists and try again.")
        return False
    
    # Initialize service with configuration
    config = create_config(quick)
    service = VideoService(config)
    
    # Validate video format
    if not service.validate_video_file(video_path):
        print(f"❌ Error: Unsupported video format")
        print(f"📋 Supported formats: {service.get_supported_formats()}")
        return False
    
    try:
        print(f"\n🔍 Starting analysis...")
        print("⏳ This may take a moment depending on video length...")
        
        # Analyze the video
        result = await service.analyze_video_file(video_path, user_prompt=prompt)
        
        if result.success:
            print_results_summary(result)
            print_workflow_steps(result.workflow_steps)
            
            # Export to browser-use format
            print("🔄 Exporting to browser-use format...")
            workflow = await service.export_workflow_to_browser_use(result.analysis_id)
            
            return save_and_show_workflow(workflow, video_path)
        else:
            print(f"❌ Analysis failed: {result.error_message}")
            return False
    
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print_error_tips(verbose)
        return False


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test video-use library with a real video file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_real_video.py my_recording.mp4
  python test_real_video.py login_demo.mp4 --prompt "User logging into website"
  python test_real_video.py long_video.mp4 --quick
  python test_real_video.py form_demo.mp4 --prompt "Fill out contact form" --verbose
        """
    )
    
    parser.add_argument('video_path', type=str, help='Path to the video file to analyze')
    parser.add_argument('--prompt', '-p', type=str, default=None, help='User prompt to provide context for the analysis')
    parser.add_argument('--quick', '-q', action='store_true', help='Use quick analysis mode (faster but less accurate)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging output')
    
    args = parser.parse_args()
    
    # Setup logging and run analysis
    setup_logging(args.verbose)
    video_path = Path(args.video_path)
    
    success = asyncio.run(analyze_real_video(
        video_path=video_path,
        prompt=args.prompt,
        quick=args.quick,
        verbose=args.verbose
    ))
    
    if success:
        print(f"\n🎉 Test completed successfully!")
        print(f"📁 Results saved in current directory")
    else:
        print(f"\n❌ Test failed. See error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main() 