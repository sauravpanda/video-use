"""
Example demonstrating complete workflow execution pipeline.
Analyzes video, generates workflow, and executes it with browser-use agent.

Usage:
    python workflow_execution_example.py <video_path>
    python workflow_execution_example.py sample_form_filling.mp4
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path

from video_use import VideoUseService, VideoAnalysisConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main(video_path: Path):
    """Demonstrate complete workflow execution pipeline."""
    
    print("🎬 Video-Use: Complete Workflow Execution Demo")
    print("=" * 50)
    
    # Check if video file exists
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        print("💡 Please provide a valid video file path")
        return
    
    # Check for required API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️ GOOGLE_API_KEY environment variable not set")
        print("💡 Set your Gemini API key: export GOOGLE_API_KEY='your-key'")
        return
    
    # Configuration
    config = VideoAnalysisConfig(
        frame_extraction_fps=1.0,
        max_frames=20  # Keep it small for demo
    )
    
    service = VideoUseService(config)
    
    print(f"\n🔄 Analyzing video: {video_path}")
    print("=" * 30)
    
    try:
        # Execute complete pipeline: analyze -> generate -> execute
        results = await service.analyze_and_execute_workflow(
            video_path=video_path,
            use_gemini=True,
            gemini_api_key=api_key,
            headless=False,  # Set to True for headless execution
        )
        
        print("\n📊 Pipeline Results:")
        print("=" * 20)
        
        # Analysis results
        analysis = results.get("analysis")
        if analysis and analysis.success:
            print("✅ Step 1: Video Analysis - SUCCESS")
            print(f"   - Analysis ID: {analysis.analysis_id}")
            print(f"   - Workflow steps: {len(analysis.workflow_steps)}")
            print(f"   - Confidence: {analysis.confidence_score:.2f}")
        else:
            print("❌ Step 1: Video Analysis - FAILED")
            if analysis:
                print(f"   - Error: {analysis.error_message}")
        
        # Workflow generation results
        workflow = results.get("workflow")
        if workflow:
            print("✅ Step 2: Workflow Generation - SUCCESS")
            print(f"   - Prompt: {workflow.prompt[:100]}...")
            print(f"   - Start URL: {workflow.start_url}")
            print(f"   - Parameters: {len(workflow.parameters)}")
            if workflow.token_usage:
                print(f"   - Tokens used: {workflow.token_usage.total_tokens}")
        else:
            print("❌ Step 2: Workflow Generation - FAILED")
        
        # Execution results
        execution = results.get("execution")
        if execution:
            if execution.success:
                print("✅ Step 3: Workflow Execution - SUCCESS")
                print(f"   - Execution ID: {execution.execution_id}")
                print(f"   - Execution time: {execution.execution_time:.2f}s")
                print(f"   - Results: {len(execution.results)} items")
            else:
                print("❌ Step 3: Workflow Execution - FAILED")
                print(f"   - Error: {execution.error_message}")
        else:
            print("❌ Step 3: Workflow Execution - FAILED")
        
        # Overall result
        print(f"\n🎯 Overall Pipeline Result: {'SUCCESS' if results.get('success') else 'FAILED'}")
        
        if results.get('error'):
            print(f"❌ Pipeline Error: {results['error']}")
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        logger.exception("Pipeline execution error")


async def demonstrate_individual_steps(video_path: Path):
    """Demonstrate individual pipeline steps for more control."""
    
    print("\n🔧 Individual Steps Demo")
    print("=" * 25)
    
    config = VideoAnalysisConfig(frame_extraction_fps=1.0, max_frames=10)
    service = VideoUseService(config)
    
    try:
        # Step 1: Analyze video
        print("Step 1: Analyzing video...")
        analysis_result = await service.analyze_video_file(
            video_path,
            use_gemini=True
        )
        
        if not analysis_result.success:
            print(f"Analysis failed: {analysis_result.error_message}")
            return
        
        print(f"✅ Analysis complete - {len(analysis_result.workflow_steps)} steps")
        
        # Step 2: Generate workflow
        print("Step 2: Generating workflow...")
        analysis_text = analysis_result.workflow_steps[0].get('analysis_text', '')
        workflow = await service.generate_structured_workflow_from_gemini(
            analysis_text,
            start_url="https://example.com"
        )
        
        print(f"✅ Workflow generated: {workflow.prompt}\n\n-------")
        
        # Step 3: Execute workflow
        print("Step 3: Executing workflow...")
        execution_result = await service.execute_workflow(
            workflow,
            headless=True,  # Headless for demo
            timeout=30
        )
        
        if execution_result.success:
            print(f"✅ Execution complete in {execution_result.execution_time:.2f}s")
        else:
            print(f"❌ Execution failed: {execution_result.error_message}")
    
    except Exception as e:
        print(f"❌ Individual steps demo failed: {e}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demonstrate complete workflow execution pipeline with video-use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python workflow_execution_example.py sample_form_filling.mp4
  python workflow_execution_example.py /path/to/your/video.mp4
  python workflow_execution_example.py login_demo.mov

Requirements:
  - Set GOOGLE_API_KEY environment variable
  - Video file in supported format (MP4, AVI, MOV, MKV, WebM)
        """
    )
    
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to the video file to analyze'
    )
    
    parser.add_argument(
        '--mode',
        choices=['complete', 'individual'],
        default='complete',
        help='Demo mode: complete pipeline or individual steps (default: complete)'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run browser in headless mode'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    video_path = Path(args.video_path)
    
    print("🚀 Video-Use Workflow Execution Demo")
    print(f"📹 Video: {video_path}")
    print(f"🎛️ Mode: {args.mode}")
    print(f"🔍 Headless: {args.headless}")
    print()
    
    if args.mode == 'complete':
        asyncio.run(main(video_path))
    else:
        asyncio.run(demonstrate_individual_steps(video_path)) 