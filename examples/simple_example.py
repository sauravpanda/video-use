"""
Simple example demonstrating the new video-use structure.
Clean and minimal implementation showcasing key features.
"""

import asyncio
from pathlib import Path

from video_use import (
    VideoUseService, VideoAnalysisConfig
)


async def main():
    """Demonstrate the new video-use structure."""
    
    # Example video path (replace with your own)
    video_path = Path("sample_form_filling.mp4")
    
    print("🎬 Video-Use: New Structure Demo")
    print("=" * 40)
    
    # 1. Basic video analysis
    print("\n1️⃣ Gemini AI Analysis + Structured Output")
    print("-" * 25)
    
    config = VideoAnalysisConfig(
        frame_extraction_fps=1.0,
        max_frames=20  # Keep it small for demo
    )
    
    service = VideoUseService(config)
    
    try:
                 
         print("\n1 Gemini AI Analysis + Structured Output")
         print("-" * 42)
         
         try:
             gemini_result = await service.analyze_video_file(
                 video_path,
                 use_gemini=True
             )
             
             if gemini_result.success:
                 print("✅ Gemini analysis complete!")
                 analysis_text = gemini_result.workflow_steps[0].get('analysis_text', '')
                 print(f"   Analysis preview: {analysis_text}...")
                 
                 # Convert Gemini analysis to structured output
                 print("\n🔄 Converting to structured output...")
                 structured_output = await service.generate_structured_workflow_from_gemini(
                     analysis_text,
                     start_url="https://example.com"
                 )
                 
                 print("✅ Structured output from Gemini!")
                 print(f"   Prompt: {structured_output.prompt}...")
                 print(f"   Start URL: {structured_output.start_url}")
                 print(f"   Parameters: {len(structured_output.parameters)}")
                 
                 if structured_output.token_usage:
                     print(f"   Tokens used: {structured_output.token_usage.total_tokens}")
             else:
                 print("⚠️ Gemini analysis failed (API key required)")
                 
         except Exception as e:
             print(f"⚠️ Gemini analysis skipped: {e}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Note: This demo uses mock data. For real analysis, provide a valid video file.")


if __name__ == "__main__":
    asyncio.run(main()) 