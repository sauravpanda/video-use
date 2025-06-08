"""Action inference from UI element changes and frame analysis."""

import logging
from typing import List, Dict, Optional
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..schema.models import (
    Frame, UIElement, Action, ActionType, 
    VideoAnalysisConfig, UIElementType
)

logger = logging.getLogger(__name__)


class ActionInferrer:
    """Infers user actions from UI element changes and visual analysis."""
    
    def __init__(self, config: VideoAnalysisConfig):
        self.config = config
        
        # Use langchain providers if available (compatible with browser-use)
        self.llm = self._initialize_llm_provider(config.llm_model)
        self.use_langchain = True
        logger.info(f"Using langchain LLM provider: {config.llm_model}")
    
    def _initialize_llm_provider(self, model_name: str):
        """Initialize the appropriate LLM provider based on model name."""
        model_lower = model_name.lower()
        
        # OpenAI models
        if model_lower.startswith('gpt') or 'openai' in model_lower:
            return ChatOpenAI(model=model_name, temperature=0.1)
        
        # Default fallback to OpenAI
        else:
            logger.warning(f"Unknown model {model_name}, falling back to gpt-4o")
            return ChatOpenAI(model="gpt-4o", temperature=0.1)
        
    async def infer_actions(self, frames: List[Frame], ui_elements_by_frame: Dict[int, List[UIElement]]) -> List[Action]:
        """Infer user actions from frame sequence and UI elements."""
        try:
            actions = []
            
            # Analyze frame pairs for actions
            for i in range(len(frames) - 1):
                current_frame = frames[i]
                next_frame = frames[i + 1]
                
                current_elements = ui_elements_by_frame.get(current_frame.frame_number, [])
                next_elements = ui_elements_by_frame.get(next_frame.frame_number, [])
                
                # Infer actions between these frames
                frame_actions = await self._infer_actions_between_frames(
                    current_frame, next_frame, current_elements, next_elements
                )
                actions.extend(frame_actions)
            
            # Post-process and clean up actions
            actions = self._post_process_actions(actions)
            
            # Use LLM to enhance action descriptions if available
            if self.config.generate_descriptions and (
                (self.use_langchain and self.llm) or 
                (not self.use_langchain and self.client)
            ):
                actions = await self._enhance_actions_with_llm(actions, frames)
            
            logger.info(f"Inferred {len(actions)} actions from {len(frames)} frames")
            return actions
            
        except Exception as e:
            logger.error(f"Error inferring actions: {e}")
            return []
    
    async def _infer_actions_between_frames(
        self, 
        current_frame: Frame, 
        next_frame: Frame,
        current_elements: List[UIElement],
        next_elements: List[UIElement]
    ) -> List[Action]:
        """Infer actions between two consecutive frames."""
        actions = []
        
        # Detect element changes
        new_elements = self._find_new_elements(current_elements, next_elements)
        removed_elements = self._find_new_elements(next_elements, current_elements)
        
        # Infer click actions from element appearances/disappearances
        click_actions = self._infer_click_actions(
            current_frame, next_frame, new_elements, removed_elements
        )
        actions.extend(click_actions)
        
        # Infer typing actions from text changes
        typing_actions = await self._infer_typing_actions(
            current_frame, next_frame, current_elements, next_elements
        )
        actions.extend(typing_actions)
        
        # Infer navigation actions from major visual changes
        nav_actions = self._infer_navigation_actions(current_frame, next_frame)
        actions.extend(nav_actions)
        
        # Infer scroll actions from element position changes
        scroll_actions = self._infer_scroll_actions(current_elements, next_elements)
        actions.extend(scroll_actions)
        
        return actions
    
    def _infer_click_actions(
        self,
        current_frame: Frame,
        next_frame: Frame, 
        new_elements: List[UIElement],
        removed_elements: List[UIElement]
    ) -> List[Action]:
        """Infer click actions from element changes."""
        actions = []
        
        # Only look for actual clickable elements that disappeared (likely clicked)
        for element in removed_elements:
            if element.element_type in [UIElementType.BUTTON, UIElementType.LINK]:
                # Only if it's a significant element (not a dropdown option)
                if element.text and len(element.text) > 2:  # Avoid single letters/short options
                    # Calculate click coordinates (center of element)
                    x, y, w, h = element.bbox
                    click_x = x + w // 2
                    click_y = y + h // 2
                    
                    action = Action(
                        action_type=ActionType.CLICK,
                        target_element=element,
                        start_frame=current_frame.frame_number,
                        end_frame=next_frame.frame_number,
                        timestamp=current_frame.timestamp,
                        coordinates=(click_x, click_y),
                        confidence=element.confidence * 0.8,
                        description=f"Click on {element.text}"
                    )
                    actions.append(action)
        
        # Only detect major UI changes (not dropdown options appearing)
        significant_new_elements = [
            e for e in new_elements 
            if e.element_type in [UIElementType.DIALOG, UIElementType.MENU] 
            and len([x for x in new_elements if x.element_type == e.element_type]) < 3  # Not many similar elements
        ]
        
        for element in significant_new_elements:
            action = Action(
                action_type=ActionType.CLICK,
                start_frame=current_frame.frame_number,
                end_frame=next_frame.frame_number,
                timestamp=current_frame.timestamp,
                confidence=0.6,
                description=f"Action triggered {element.element_type.value}"
            )
            actions.append(action)
        
        return actions
    
    async def _infer_typing_actions(
        self,
        current_frame: Frame,
        next_frame: Frame,
        current_elements: List[UIElement],
        next_elements: List[UIElement]
    ) -> List[Action]:
        """Infer typing actions from text changes in input fields."""
        actions = []
        
        # Find input fields in both frames
        current_inputs = [e for e in current_elements if e.element_type == UIElementType.INPUT]
        next_inputs = [e for e in next_elements if e.element_type == UIElementType.INPUT]
        
        # Match input fields between frames
        for current_input in current_inputs:
            matching_input = self._find_matching_element(current_input, next_inputs)
            
            if matching_input:
                current_text = current_input.text or ""
                next_text = matching_input.text or ""
                
                # Check if text changed
                if current_text != next_text:
                    typed_text = next_text
                    if current_text and next_text.startswith(current_text):
                        # New text was added
                        typed_text = next_text[len(current_text):]
                    
                    action = Action(
                        action_type=ActionType.TYPE,
                        target_element=matching_input,
                        value=typed_text,
                        start_frame=current_frame.frame_number,
                        end_frame=next_frame.frame_number,
                        timestamp=current_frame.timestamp,
                        confidence=0.8,
                        description=f"Type '{typed_text}' in input field"
                    )
                    actions.append(action)
        
        return actions
    
    def _infer_navigation_actions(self, current_frame: Frame, next_frame: Frame) -> List[Action]:
        """Infer navigation actions from major visual changes."""
        actions = []
        
        # Check if this looks like a page navigation (lowered threshold for browser interactions)
        if next_frame.visual_diff_score > 0.15:  # More sensitive to detect navigation/page changes
            action = Action(
                action_type=ActionType.NAVIGATE,
                start_frame=current_frame.frame_number,
                end_frame=next_frame.frame_number,
                timestamp=current_frame.timestamp,
                confidence=min(0.7, next_frame.visual_diff_score),
                description="Navigate to new page"
            )
            actions.append(action)
        
        return actions
    
    def _infer_scroll_actions(
        self, 
        current_elements: List[UIElement], 
        next_elements: List[UIElement]
    ) -> List[Action]:
        """Infer scroll actions from element position changes."""
        actions = []
        
        # Look for consistent vertical displacement of elements
        displacements = []
        
        for current_elem in current_elements:
            matching_elem = self._find_matching_element(current_elem, next_elements)
            if matching_elem:
                current_y = current_elem.bbox[1]
                next_y = matching_elem.bbox[1]
                displacement = next_y - current_y
                
                if abs(displacement) > 10:  # Significant movement
                    displacements.append(displacement)
        
        if len(displacements) >= 3:  # Multiple elements moved
            avg_displacement = np.mean(displacements)
            
            if abs(avg_displacement) > 20:  # Significant average movement
                scroll_direction = "down" if avg_displacement < 0 else "up"
                
                action = Action(
                    action_type=ActionType.SCROLL,
                    value=scroll_direction,
                    confidence=0.7,
                    description=f"Scroll {scroll_direction}"
                )
                actions.append(action)
        
        return actions
    
    def _find_matching_element(self, element: UIElement, candidates: List[UIElement]) -> Optional[UIElement]:
        """Find a matching element in the candidates list."""
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            if candidate.element_type != element.element_type:
                continue
            
            # Calculate similarity score
            score = self._calculate_element_similarity(element, candidate)
            
            if score > best_score and score > 0.5:
                best_score = score
                best_match = candidate
        
        return best_match
    
    def _calculate_element_similarity(self, elem1: UIElement, elem2: UIElement) -> float:
        """Calculate similarity between two UI elements."""
        # Position similarity
        x1, y1, w1, h1 = elem1.bbox
        x2, y2, w2, h2 = elem2.bbox
        
        # Calculate IoU (Intersection over Union)
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left < right and top < bottom:
            intersection = (right - left) * (bottom - top)
            union = w1 * h1 + w2 * h2 - intersection
            iou = intersection / union if union > 0 else 0
        else:
            iou = 0
        
        # Text similarity
        text_similarity = 0.0
        if elem1.text and elem2.text:
            # Simple text similarity (could be improved with fuzzy matching)
            if elem1.text == elem2.text:
                text_similarity = 1.0
            elif elem1.text in elem2.text or elem2.text in elem1.text:
                text_similarity = 0.8
        elif not elem1.text and not elem2.text:
            text_similarity = 1.0
        
        # Combined similarity (weighted average)
        similarity = 0.7 * iou + 0.3 * text_similarity
        return similarity
    
    def _find_new_elements(self, old_elements: List[UIElement], new_elements: List[UIElement]) -> List[UIElement]:
        """Find elements that are in new_elements but not in old_elements."""
        new = []
        
        for new_elem in new_elements:
            if not self._find_matching_element(new_elem, old_elements):
                new.append(new_elem)
        
        return new
    
    def _post_process_actions(self, actions: List[Action]) -> List[Action]:
        """Post-process actions to remove duplicates and improve quality."""
        if not actions:
            return actions
        
        # Sort actions by timestamp
        actions.sort(key=lambda a: a.timestamp)
        
        # Remove duplicate actions
        filtered_actions = []
        for action in actions:
            if not self._is_duplicate_action(action, filtered_actions):
                filtered_actions.append(action)
        
        # Filter out likely false positives (dropdown options, etc.)
        quality_filtered_actions = []
        for action in filtered_actions:
            if self._is_likely_real_action(action):
                quality_filtered_actions.append(action)
        
        # Filter by confidence threshold
        high_confidence_actions = [
            action for action in quality_filtered_actions 
            if action.confidence >= self.config.action_confidence_threshold
        ]
        
        return high_confidence_actions
    
    def _is_duplicate_action(self, action: Action, existing_actions: List[Action]) -> bool:
        """Check if an action is a duplicate of existing actions."""
        for existing in existing_actions:
            # Same action type and similar timing
            if (action.action_type == existing.action_type and
                abs(action.timestamp - existing.timestamp) < 0.5):
                
                # Check if they target similar elements or positions
                if action.target_element and existing.target_element:
                    similarity = self._calculate_element_similarity(
                        action.target_element, existing.target_element
                    )
                    if similarity > 0.8:
                        return True
                elif action.coordinates and existing.coordinates:
                    # Check coordinate similarity
                    distance = np.sqrt(
                        (action.coordinates[0] - existing.coordinates[0]) ** 2 +
                        (action.coordinates[1] - existing.coordinates[1]) ** 2
                    )
                    if distance < 20:  # Within 20 pixels
                        return True
        
        return False
    
    def _is_likely_real_action(self, action: Action) -> bool:
        """Check if an action is likely a real user action (not a false positive)."""
        # Skip actions with very generic descriptions
        if action.description:
            generic_phrases = [
                "Action triggered menu",
                "Action triggered dialog", 
                "Click on and",
                "Click on say",
                "Click on "
            ]
            
            # If description is too generic or very short, likely false positive
            if any(phrase in action.description for phrase in generic_phrases):
                return False
            
            # If clicking on single letters or very short text, likely dropdown option
            if action.target_element and action.target_element.text:
                text = action.target_element.text.strip()
                if len(text) <= 2 and text.isalpha():  # Single letters like "CA", "SF"
                    return False
        
        # Navigation actions should have significant visual change
        if action.action_type == ActionType.NAVIGATE:
            return action.confidence > 0.5
        
        # Type actions should have actual content
        if action.action_type == ActionType.TYPE:
            return action.value and len(action.value.strip()) > 0
        
        # Click actions should target meaningful elements
        if action.action_type == ActionType.CLICK:
            if action.target_element and action.target_element.text:
                text = action.target_element.text.strip()
                # Accept clicks on buttons/links with meaningful text
                return len(text) > 2 or text.lower() in ['ok', 'yes', 'no', 'go']
            
            # If no target element, be more conservative
            return action.confidence > 0.7
        
        return True
    
    async def _enhance_actions_with_llm(self, actions: List[Action], frames: List[Frame]) -> List[Action]:
        """Use LLM to enhance action descriptions and context."""
        if self.use_langchain and not self.llm:
            return actions
        elif not self.use_langchain and not self.client:
            return actions
        
        try:
            # Create context for LLM
            context = self._create_llm_context(actions, frames)
            
            prompt = f"""
            Analyze the following browser actions extracted from a video and provide enhanced descriptions:
            
            Context: {context}
            
            Actions:
            {self._format_actions_for_llm(actions)}
            
            Please provide:
            1. Enhanced descriptions for each action
            2. Identify the overall workflow/task being performed
            3. Suggest any missing actions that might be implied
            
            Respond in JSON format with enhanced action descriptions.
            """
            
            if self.use_langchain:
                # Use langchain provider
                messages = [
                    SystemMessage(content="You are an AI assistant that analyzes browser automation workflows."),
                    HumanMessage(content=prompt)
                ]
                response = await self.llm.ainvoke(messages)
                response_content = response.content
            else:
                # Use direct OpenAI client as fallback
                response = await self.client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                response_content = response.choices[0].message.content
            
            # Parse response and update actions
            enhanced_descriptions = self._parse_llm_response(response_content)
            
            for i, action in enumerate(actions):
                if i < len(enhanced_descriptions):
                    action.description = enhanced_descriptions[i]
            
            return actions
            
        except Exception as e:
            logger.error(f"Error enhancing actions with LLM: {e}")
            return actions
    
    def _create_llm_context(self, actions: List[Action], frames: List[Frame]) -> str:
        """Create context string for LLM analysis."""
        context_parts = [
            f"Video duration: {frames[-1].timestamp - frames[0].timestamp:.1f} seconds",
            f"Total frames analyzed: {len(frames)}",
            f"Actions detected: {len(actions)}"
        ]
        
        # Add action types summary
        action_types = {}
        for action in actions:
            action_types[action.action_type.value] = action_types.get(action.action_type.value, 0) + 1
        
        if action_types:
            context_parts.append(f"Action types: {action_types}")
        
        return "; ".join(context_parts)
    
    def _format_actions_for_llm(self, actions: List[Action]) -> str:
        """Format actions for LLM analysis."""
        formatted = []
        for i, action in enumerate(actions):
            action_str = f"{i+1}. {action.action_type.value}"
            if action.target_element and action.target_element.text:
                action_str += f" on '{action.target_element.text}'"
            if action.value:
                action_str += f" with value '{action.value}'"
            action_str += f" (confidence: {action.confidence:.2f})"
            formatted.append(action_str)
        
        return "\n".join(formatted)
    
    def _parse_llm_response(self, response: str) -> List[str]:
        """Parse LLM response to extract enhanced descriptions."""
        # This is a simplified parser - you'd want more robust JSON parsing
        try:
            import json
            data = json.loads(response)
            if isinstance(data, dict) and 'descriptions' in data:
                return data['descriptions']
            elif isinstance(data, list):
                return data
        except:
            pass
        
        # Fallback: return original response split by lines
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        return lines[:10]  # Limit to reasonable number 