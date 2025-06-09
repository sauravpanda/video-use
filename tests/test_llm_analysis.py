"""
Test script for the new LLM-based sequential frame analysis.

This script tests the modified video analysis logic that:
1. Selects every 15th frame
2. Analyzes each frame sequentially with LLM
3. Builds workflow based on LLM analysis of each frame
"""

import asyncio
import logging
import pytest
from pathlib import Path
from video_use import VideoUseService, VideoAnalysisConfig

# Enable logging to see what's happening
logging.basicConfig(level=logging.INFO)

@pytest.mark.asyncio
async def test_llm_analysis():
    """Test the new LLM-based frame analysis."""
    
    print("🎥 Testing LLM-Based Sequential Frame Analysis")
    print("=" * 60)

    # Look for any video file in examples directory
    video_files = list(Path("examples").glob("*.mp4"))
    if not video_files:
        print("❌ No video files found in examples directory")
        print("Please add a video file to the examples directory to test")
        pytest.skip("No video files found in examples directory")
        return
    
    video_path = video_files[0]
    print(f"📹 Using video file: {video_path}")
    
    # Configuration for LLM analysis
    config = VideoAnalysisConfig(
        frame_extraction_fps=1.0,      # Will be overridden to use every 15th frame
        ui_detection_confidence=0.7,   
        enable_ocr=True,               
        max_frames=10,                 # Limit for testing
        parallel_processing=True,       
        generate_descriptions=True,    # Enable LLM descriptions
        llm_model="gpt-4o"            # Use GPT-4 for vision analysis
    )
    
    # Initialize the video service
    service = VideoUseService(config)
    
    try:
        print(f"🔍 Starting LLM-based analysis...")
        print(f"📊 Configuration:")
        print(f"   - Frame selection: Every 15th frame")
        print(f"   - Max frames: {config.max_frames}")
        print(f"   - LLM model: {config.llm_model}")
        print(f"   - Sequential analysis with step context")
        print()
        
        # Analyze the video with the new approach
        result = await service.analyze_video_file(
            video_path,
            user_prompt="Analyze this browser workflow step by step"
        )
        
        if result.success:
            print(f"✅ LLM Analysis completed successfully!")
            print(f"📊 Results:")
            print(f"   - Analysis ID: {result.analysis_id}")
            print(f"   - Processing time: {result.processing_time:.2f} seconds")
            print(f"   - Confidence score: {result.confidence_score:.2f}")
            print(f"   - Workflow steps found: {len(result.workflow_steps)}")
            print()
            
            # Display workflow steps with LLM descriptions
            if result.workflow_steps:
                print(f"🔄 LLM-Analyzed Workflow Steps:")
                print("-" * 50)
                for i, step in enumerate(result.workflow_steps, 1):
                    print(f"Step {i}: {step['action_type'].upper()}")
                    print(f"   Description: {step['description']}")
                    print(f"   Timestamp: {step['timestamp']:.2f}s")
                    print(f"   Frame: {step['frame_number']}")
                    print(f"   Confidence: {step['confidence']:.2f}")
                    
                    if 'target_element' in step and step['target_element']:
                        element = step['target_element']
                        if 'text' in element and element['text']:
                            print(f"   Target Element: {element['text']}")
                    
                    print()
            
            # Show the difference with previous approach
            print("🔍 Analysis Method Differences:")
            print(f"   - Frame Selection: Every 15th frame (vs dynamic FPS-based)")
            print(f"   - Analysis: Sequential LLM evaluation (vs heuristic rules)")
            print(f"   - Context: Previous steps shared with LLM (vs isolated analysis)")
            print(f"   - Confidence: LLM-based assessment (vs rule-based scoring)")
            
        else:
            print(f"❌ Analysis failed: {result.error_message}")
    
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

# Test can be run with: pytest tests/test_llm_analysis.py -v 