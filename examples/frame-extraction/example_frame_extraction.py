"""
Clean example for video analysis using the video_use library.

This example demonstrates:
1. Frame extraction using FrameExtractor
2. Video analysis using GeminiVideoAnalyzer
"""

import asyncio
import argparse
from pathlib import Path

from video_use.video import FrameExtractor, GeminiVideoAnalyzer
from video_use.schema.models import VideoAnalysisConfig


async def extract_frames_example(video_path: Path):
    """Extract frames from video and save them."""
    print(f"🔍 Extracting frames from: {video_path}")
    
    # Configure frame extraction (1 frame per 30 source frames)
    config = VideoAnalysisConfig(
        min_frame_difference=0.02,
        fps=1.0  # Extract roughly 1 frame per second
    )
    
    # Extract frames
    extractor = FrameExtractor(config)
    frames = await extractor.extract_frames(video_path)
    
    print(f"✅ Extracted {len(frames)} frames")
    for i, frame in enumerate(frames[:5]):  # Show first 5
        print(f"   Frame {i+1}: #{frame.frame_number} at {frame.timestamp:.2f}s")
    
    if len(frames) > 5:
        print(f"   ... and {len(frames) - 5} more frames")


async def analyze_with_gemini(video_path: Path):
    """Analyze video using Gemini API."""
    print(f"🤖 Analyzing video with Gemini: {video_path}")
    
    try:
        analyzer = GeminiVideoAnalyzer()
        result = await analyzer.analyze_video(video_path)
        
        if result["success"]:
            print("✅ Analysis complete!")
            print("\n" + "="*60)
            print("STEP-BY-STEP USER ACTIONS:")
            print("="*60)
            print(result["analysis"])
        else:
            print(f"❌ Analysis failed: {result['error']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Video analysis example")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--mode", choices=["frames", "gemini"], default="frames",
                       help="Analysis mode: 'frames' for frame extraction, 'gemini' for AI analysis")
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return
    
    if args.mode == "frames":
        await extract_frames_example(video_path)
    elif args.mode == "gemini":
        await analyze_with_gemini(video_path)


if __name__ == "__main__":
    asyncio.run(main()) 