"""Main video analyzer that orchestrates all video processing components."""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..schema.models import (
    VideoAnalysisResult, VideoAnalysisConfig, Frame, UIElement, 
    Action, WorkflowDefinition, VideoWorkflowStep
)
from .frame_extractor import FrameExtractor
from .ui_detector import UIDetector
from .action_inferrer import ActionInferrer
from .ocr_service import OCRService

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """Main video analyzer that coordinates all processing components."""
    
    def __init__(self, config: Optional[VideoAnalysisConfig] = None):
        self.config = config or VideoAnalysisConfig()
        
        # Initialize components
        self.frame_extractor = FrameExtractor(self.config)
        self.ui_detector = UIDetector(self.config)
        self.action_inferrer = ActionInferrer(self.config)
        self.ocr_service = OCRService(self.config)
        
        logger.info("VideoAnalyzer initialized with config")
    
    async def analyze_video(self, video_path: Path) -> VideoAnalysisResult:
        """
        Analyze a video file and extract workflow actions.
        
        This is the main entry point that:
        1. Extracts frames from video
        2. Detects UI elements in frames 
        3. Infers user actions from changes
        4. Generates workflow definition
        
        Args:
            video_path: Path to the video file to analyze
            
        Returns:
            VideoAnalysisResult containing all extracted information
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting video analysis for: {video_path}")
            
            # Validate input
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Extract video metadata
            video_metadata = await self.frame_extractor.extract_video_metadata(video_path)
            logger.info(f"Video metadata: {video_metadata.duration:.1f}s, {video_metadata.fps:.1f}fps")
            
            # Extract frames
            frames = await self.frame_extractor.extract_frames(video_path)
            logger.info(f"Extracted {len(frames)} frames")
            
            if not frames:
                return VideoAnalysisResult(
                    video_metadata=video_metadata,
                    frames=[],
                    ui_elements=[],
                    actions=[],
                    success=False,
                    error_message="No frames could be extracted from video"
                )
            
            # Detect UI elements in all frames
            ui_elements_by_frame = await self._detect_ui_elements_in_frames(frames)
            all_ui_elements = []
            for elements in ui_elements_by_frame.values():
                all_ui_elements.extend(elements)
            
            logger.info(f"Detected {len(all_ui_elements)} UI elements across all frames")
            
            # Infer actions from frame and UI element changes
            actions = await self.action_inferrer.infer_actions(frames, ui_elements_by_frame)
            logger.info(f"Inferred {len(actions)} actions")
            
            # Generate workflow definition
            workflow = await self._generate_workflow(actions, frames, video_metadata)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            result = VideoAnalysisResult(
                video_metadata=video_metadata,
                frames=frames,
                ui_elements=all_ui_elements,
                actions=actions,
                workflow=workflow,
                processing_time=processing_time,
                success=True
            )
            
            logger.info(f"Video analysis completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Video analysis failed: {e}")
            
            return VideoAnalysisResult(
                video_metadata=video_metadata if 'video_metadata' in locals() else None,
                frames=frames if 'frames' in locals() else [],
                ui_elements=[],
                actions=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def _detect_ui_elements_in_frames(self, frames: List[Frame]) -> Dict[int, List[UIElement]]:
        """Detect UI elements in all frames in parallel."""
        ui_elements_by_frame = {}
        
        if self.config.parallel_processing:
            # Process frames in parallel batches
            batch_size = min(self.config.max_workers, len(frames))
            
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]
                
                # Create tasks for this batch
                tasks = []
                for frame in batch:
                    task = self.ui_detector.detect_ui_elements(frame)
                    tasks.append((frame.frame_number, task))
                
                # Wait for batch to complete
                batch_results = await asyncio.gather(*[task for _, task in tasks])
                
                # Store results
                for (frame_number, _), elements in zip(tasks, batch_results):
                    ui_elements_by_frame[frame_number] = elements
                
                logger.debug(f"Processed UI detection batch {i//batch_size + 1}/{(len(frames) + batch_size - 1)//batch_size}")
        else:
            # Process frames sequentially
            for frame in frames:
                elements = await self.ui_detector.detect_ui_elements(frame)
                ui_elements_by_frame[frame.frame_number] = elements
        
        return ui_elements_by_frame
    
    async def _generate_workflow(
        self, 
        actions: List[Action], 
        frames: List[Frame],
        video_metadata
    ) -> Optional[WorkflowDefinition]:
        """Generate a workflow definition from the extracted actions."""
        try:
            if not actions:
                logger.warning("No actions found, cannot generate workflow")
                return None
            
            # Create workflow steps from actions
            workflow_steps = []
            
            for i, action in enumerate(actions):
                step_id = f"step_{i+1}"
                
                # Find the frame this action occurred in
                action_frame = None
                for frame in frames:
                    if frame.frame_number == action.start_frame:
                        action_frame = frame
                        break
                
                # Create workflow step
                step = VideoWorkflowStep(
                    step_id=step_id,
                    action=action,
                    description=action.description or f"{action.action_type.value} action",
                    video_timestamp=action.timestamp,
                    confidence_score=action.confidence,
                    frame_number=action.start_frame,
                    screenshot_path=action_frame.image_path if action_frame else None
                )
                
                workflow_steps.append(step)
            
            # Calculate overall confidence
            if workflow_steps:
                avg_confidence = sum(step.confidence_score for step in workflow_steps) / len(workflow_steps)
            else:
                avg_confidence = 0.0
            
            # Estimate duration
            estimated_duration = video_metadata.duration if video_metadata else 0.0
            
            # Generate workflow name and description
            workflow_name = self._generate_workflow_name(actions)
            workflow_description = self._generate_workflow_description(actions, workflow_steps)
            
            workflow = WorkflowDefinition(
                name=workflow_name,
                description=workflow_description,
                steps=workflow_steps,
                estimated_duration=estimated_duration,
                confidence_score=avg_confidence,
                metadata={
                    'video_file': str(video_metadata.file_path) if video_metadata else None,
                    'total_frames': len(frames),
                    'analysis_config': {
                        'frame_extraction_fps': self.config.frame_extraction_fps,
                        'ui_detection_confidence': self.config.ui_detection_confidence,
                        'action_confidence_threshold': self.config.action_confidence_threshold
                    }
                }
            )
            
            logger.info(f"Generated workflow '{workflow_name}' with {len(workflow_steps)} steps")
            return workflow
            
        except Exception as e:
            logger.error(f"Error generating workflow: {e}")
            return None
    
    def _generate_workflow_name(self, actions: List[Action]) -> str:
        """Generate a descriptive name for the workflow."""
        if not actions:
            return "Empty Workflow"
        
        # Count action types
        action_counts = {}
        for action in actions:
            action_type = action.action_type.value
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        # Find most common action
        most_common_action = max(action_counts.items(), key=lambda x: x[1])[0]
        
        # Generate name based on actions
        if 'type' in action_counts and 'click' in action_counts:
            return "Form Filling Workflow"
        elif 'navigate' in action_counts:
            return "Navigation Workflow"
        elif 'click' in action_counts:
            return "Button Interaction Workflow"
        elif 'scroll' in action_counts:
            return "Page Browsing Workflow"
        else:
            return f"{most_common_action.title()} Workflow"
    
    def _generate_workflow_description(self, actions: List[Action], steps: List[VideoWorkflowStep]) -> str:
        """Generate a description for the workflow."""
        if not actions:
            return "No actions detected in video"
        
        # Count different types of actions
        action_counts = {}
        for action in actions:
            action_type = action.action_type.value
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        # Build description
        desc_parts = [f"Workflow with {len(steps)} steps"]
        
        if action_counts:
            action_summary = []
            for action_type, count in action_counts.items():
                if count == 1:
                    action_summary.append(f"1 {action_type}")
                else:
                    action_summary.append(f"{count} {action_type} actions")
            
            desc_parts.append(f"including {', '.join(action_summary)}")
        
        # Add timing info
        if actions:
            duration = actions[-1].timestamp - actions[0].timestamp
            desc_parts.append(f"Duration: {duration:.1f} seconds")
        
        return ". ".join(desc_parts) + "."
    
    async def analyze_video_with_prompt(self, video_path: Path, user_prompt: str) -> VideoAnalysisResult:
        """
        Analyze video with additional context from user prompt.
        
        This method can be enhanced to use the user prompt to:
        - Focus on specific types of actions
        - Improve action inference with context
        - Generate more targeted workflow descriptions
        """
        # For now, perform standard analysis
        # TODO: Integrate user prompt into analysis pipeline
        result = await self.analyze_video(video_path)
        
        if result.workflow:
            # Enhance workflow description with user context
            result.workflow.description += f" User context: {user_prompt}"
            result.workflow.metadata['user_prompt'] = user_prompt
        
        return result
    
    async def extract_keyframe_actions(self, video_path: Path) -> VideoAnalysisResult:
        """
        Analyze video using only keyframes for faster processing.
        
        This is useful for quick analysis or when processing resources are limited.
        """
        # Temporarily adjust config for keyframe-only analysis
        original_fps = self.config.frame_extraction_fps
        self.config.frame_extraction_fps = 0.5  # Lower FPS for keyframes
        
        try:
            # Extract keyframes instead of regular frames
            keyframes = await self.frame_extractor.extract_keyframes(video_path)
            
            if not keyframes:
                return VideoAnalysisResult(
                    video_metadata=await self.frame_extractor.extract_video_metadata(video_path),
                    frames=[],
                    ui_elements=[],
                    actions=[],
                    success=False,
                    error_message="No keyframes could be extracted"
                )
            
            # Process keyframes as regular frames
            ui_elements_by_frame = await self._detect_ui_elements_in_frames(keyframes)
            
            all_ui_elements = []
            for elements in ui_elements_by_frame.values():
                all_ui_elements.extend(elements)
            
            actions = await self.action_inferrer.infer_actions(keyframes, ui_elements_by_frame)
            
            video_metadata = await self.frame_extractor.extract_video_metadata(video_path)
            workflow = await self._generate_workflow(actions, keyframes, video_metadata)
            
            return VideoAnalysisResult(
                video_metadata=video_metadata,
                frames=keyframes,
                ui_elements=all_ui_elements,
                actions=actions,
                workflow=workflow,
                success=True
            )
            
        finally:
            # Restore original config
            self.config.frame_extraction_fps = original_fps
    
    async def save_analysis_results(self, result: VideoAnalysisResult, output_dir: Path) -> Path:
        """Save analysis results to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save frames if they haven't been saved yet
        if result.frames and not result.frames[0].image_path:
            frames_dir = output_dir / "frames"
            result.frames = await self.frame_extractor.save_frames(result.frames, frames_dir)
        
        # Save analysis results as JSON
        import json
        from dataclasses import asdict
        
        results_file = output_dir / "analysis_results.json"
        
        # Convert result to dictionary (simplified)
        result_dict = {
            'success': result.success,
            'processing_time': result.processing_time,
            'error_message': result.error_message,
            'video_metadata': asdict(result.video_metadata) if result.video_metadata else None,
            'frames_count': len(result.frames),
            'ui_elements_count': len(result.ui_elements),
            'actions_count': len(result.actions),
            'workflow': asdict(result.workflow) if result.workflow else None
        }
        
        with open(results_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        logger.info(f"Analysis results saved to {results_file}")
        return results_file 