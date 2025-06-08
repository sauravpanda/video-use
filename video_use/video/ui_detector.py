"""UI element detection using computer vision."""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    logging.warning("YOLO not available. Install ultralytics for enhanced UI detection.")

from ..schema.models import Frame, UIElement, UIElementType, VideoAnalysisConfig
from .ocr_service import OCRService, OCRResult

logger = logging.getLogger(__name__)


class UIDetector:
    """Detects UI elements in video frames using computer vision techniques."""
    
    def __init__(self, config: VideoAnalysisConfig):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.ocr_service = OCRService(config)
        self.yolo_model = None
        
        # Initialize YOLO if available
        if HAS_YOLO:
            try:
                # Note: You might want to use a custom trained model for UI elements
                # For now, using a general object detection model
                self.yolo_model = YOLO('yolov8n.pt')
                logger.info("YOLO model initialized for UI detection")
            except Exception as e:
                logger.warning(f"Failed to initialize YOLO: {e}")
    
    async def detect_ui_elements(self, frame: Frame) -> List[UIElement]:
        """Detect all UI elements in a frame."""
        try:
            # Get OCR results for text-based elements
            ocr_results = await self.ocr_service.extract_text_from_frame(frame)
            
            # Detect visual elements
            loop = asyncio.get_event_loop()
            visual_elements = await loop.run_in_executor(
                self.executor,
                self._detect_visual_elements,
                frame.image
            )
            
            # Combine OCR and visual detection results
            ui_elements = []
            
            # Add text-based elements from OCR
            for ocr_result in ocr_results:
                ui_type = self._classify_text_element(ocr_result.text)
                if ui_type:
                    ui_elements.append(UIElement(
                        element_type=ui_type,
                        bbox=ocr_result.bbox,
                        confidence=ocr_result.confidence,
                        text=ocr_result.text,
                        frame_number=frame.frame_number
                    ))
            
            # Add visual elements
            ui_elements.extend(visual_elements)
            
            # Filter by confidence threshold
            filtered_elements = [
                elem for elem in ui_elements 
                if elem.confidence >= self.config.ui_detection_confidence
            ]
            
            logger.debug(f"Detected {len(filtered_elements)} UI elements in frame {frame.frame_number}")
            return filtered_elements
            
        except Exception as e:
            logger.error(f"Error detecting UI elements in frame {frame.frame_number}: {e}")
            return []
    
    def _detect_visual_elements(self, image: np.ndarray) -> List[UIElement]:
        """Detect visual UI elements using computer vision."""
        elements = []
        
        # Detect buttons using color and shape analysis
        buttons = self._detect_buttons(image)
        elements.extend(buttons)
        
        # Detect input fields
        inputs = self._detect_input_fields(image)
        elements.extend(inputs)
        
        # Detect links (typically blue text)
        links = self._detect_links(image)
        elements.extend(links)
        
        # Use YOLO if available for general object detection
        if self.yolo_model:
            yolo_elements = self._detect_with_yolo(image)
            elements.extend(yolo_elements)
        
        return elements
    
    def _detect_buttons(self, image: np.ndarray) -> List[UIElement]:
        """Detect button-like elements using shape and color analysis."""
        try:
            buttons = []
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Find rectangular shapes that could be buttons
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's roughly rectangular (4 corners)
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter by size - buttons are typically medium-sized
                    if 30 < w < 300 and 20 < h < 80:
                        # Check aspect ratio - buttons are usually wider than tall
                        aspect_ratio = w / h
                        if 1.5 < aspect_ratio < 8:
                            # Calculate confidence based on rectangularity
                            area = cv2.contourArea(contour)
                            rect_area = w * h
                            rectangularity = area / rect_area if rect_area > 0 else 0
                            
                            if rectangularity > 0.7:
                                buttons.append(UIElement(
                                    element_type=UIElementType.BUTTON,
                                    bbox=(x, y, w, h),
                                    confidence=min(0.8, rectangularity),
                                    attributes={'detection_method': 'shape_analysis'}
                                ))
            
            return buttons
            
        except Exception as e:
            logger.error(f"Error detecting buttons: {e}")
            return []
    
    def _detect_input_fields(self, image: np.ndarray) -> List[UIElement]:
        """Detect input field elements."""
        try:
            inputs = []
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Input fields are often thin rectangles with borders
            # Use morphological operations to find rectangular shapes
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
            morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            edges = cv2.Canny(morphed, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Input fields are typically wide and not very tall
                if w > 100 and 15 < h < 50:
                    aspect_ratio = w / h
                    if aspect_ratio > 3:  # Wide rectangles
                        # Check if it looks like an input field
                        roi = gray[y:y+h, x:x+w]
                        if roi.size > 0:
                            # Input fields often have uniform backgrounds
                            std_dev = np.std(roi)
                            if std_dev < 30:  # Relatively uniform
                                inputs.append(UIElement(
                                    element_type=UIElementType.INPUT,
                                    bbox=(x, y, w, h),
                                    confidence=0.7,
                                    attributes={'detection_method': 'shape_analysis'}
                                ))
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error detecting input fields: {e}")
            return []
    
    def _detect_links(self, image: np.ndarray) -> List[UIElement]:
        """Detect link elements (typically blue text)."""
        try:
            links = []
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Define range for blue colors (typical link color)
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            
            # Create mask for blue colors
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Find contours in blue regions
            contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Links are typically text-sized
                if 20 < w < 200 and 10 < h < 30:
                    # Check if it's roughly text-like (wider than tall)
                    aspect_ratio = w / h
                    if aspect_ratio > 1.5:
                        links.append(UIElement(
                            element_type=UIElementType.LINK,
                            bbox=(x, y, w, h),
                            confidence=0.6,
                            attributes={'detection_method': 'color_analysis'}
                        ))
            
            return links
            
        except Exception as e:
            logger.error(f"Error detecting links: {e}")
            return []
    
    def _detect_with_yolo(self, image: np.ndarray) -> List[UIElement]:
        """Use YOLO for general object detection."""
        try:
            if not self.yolo_model:
                return []
            
            # Run YOLO inference
            results = self.yolo_model(image)
            
            elements = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                        
                        confidence = box.conf[0].item()
                        class_id = int(box.cls[0].item())
                        
                        # Map YOLO classes to UI element types (this is basic mapping)
                        ui_type = self._map_yolo_class_to_ui_type(class_id)
                        
                        if ui_type and confidence > 0.5:
                            elements.append(UIElement(
                                element_type=ui_type,
                                bbox=(x, y, w, h),
                                confidence=confidence,
                                attributes={
                                    'detection_method': 'yolo',
                                    'yolo_class_id': class_id
                                }
                            ))
            
            return elements
            
        except Exception as e:
            logger.error(f"Error with YOLO detection: {e}")
            return []
    
    def _map_yolo_class_to_ui_type(self, class_id: int) -> Optional[UIElementType]:
        """Map YOLO class IDs to UI element types."""
        # This is a basic mapping - you'd want to train a custom model
        # for better UI element detection
        
        # COCO dataset mappings (very limited for UI elements)
        if class_id == 72:  # tv/monitor
            return UIElementType.IMAGE
        elif class_id == 73:  # laptop
            return UIElementType.IMAGE
        elif class_id == 67:  # cell phone
            return UIElementType.IMAGE
        
        return None
    
    def _classify_text_element(self, text: str) -> Optional[UIElementType]:
        """Classify text as a UI element type."""
        if not text or not text.strip():
            return None
        
        text_lower = text.lower().strip()
        
        # Button patterns
        button_patterns = [
            'click', 'submit', 'send', 'save', 'cancel', 'ok', 'yes', 'no',
            'login', 'sign in', 'sign up', 'register', 'continue', 'next',
            'back', 'previous', 'finish', 'done', 'apply', 'confirm', 'search'
        ]
        
        # Link patterns
        link_patterns = [
            'http', 'www', '.com', '.org', '.net', 'click here', 'learn more',
            'read more', 'view all', 'see more'
        ]
        
        # Navigation patterns
        nav_patterns = [
            'home', 'about', 'contact', 'services', 'products', 'help',
            'settings', 'profile', 'dashboard', 'menu', 'navigation'
        ]
        
        # Form label patterns
        label_patterns = [
            'name:', 'email:', 'password:', 'username:', 'phone:', 'address:',
            'first name', 'last name', 'date of birth', 'zip code'
        ]
        
        if any(pattern in text_lower for pattern in button_patterns):
            return UIElementType.BUTTON
        elif any(pattern in text_lower for pattern in link_patterns):
            return UIElementType.LINK
        elif any(pattern in text_lower for pattern in nav_patterns):
            return UIElementType.MENU
        elif any(pattern in text_lower for pattern in label_patterns) or text.endswith(':'):
            return UIElementType.TEXT
        elif len(text.split()) <= 3 and text.isalpha():
            # Short alphabetic text might be a button
            return UIElementType.BUTTON
        
        return UIElementType.TEXT
    
    async def detect_ui_changes(self, frames: List[Frame]) -> List[Dict[str, Any]]:
        """Detect changes in UI elements between frames."""
        changes = []
        prev_elements = []
        
        for frame in frames:
            current_elements = await self.detect_ui_elements(frame)
            
            # Compare with previous frame
            if prev_elements:
                # Find new elements
                new_elements = self._find_new_elements(prev_elements, current_elements)
                
                # Find removed elements
                removed_elements = self._find_new_elements(current_elements, prev_elements)
                
                if new_elements or removed_elements:
                    changes.append({
                        'frame_number': frame.frame_number,
                        'timestamp': frame.timestamp,
                        'new_elements': len(new_elements),
                        'removed_elements': len(removed_elements),
                        'total_elements': len(current_elements),
                        'change_type': 'ui_change'
                    })
            
            prev_elements = current_elements
        
        return changes
    
    def _find_new_elements(self, old_elements: List[UIElement], new_elements: List[UIElement]) -> List[UIElement]:
        """Find elements that are in new_elements but not in old_elements."""
        new = []
        
        for new_elem in new_elements:
            found_match = False
            
            for old_elem in old_elements:
                # Check if elements are similar (same type and overlapping position)
                if (new_elem.element_type == old_elem.element_type and 
                    self._elements_overlap(new_elem, old_elem)):
                    found_match = True
                    break
            
            if not found_match:
                new.append(new_elem)
        
        return new
    
    def _elements_overlap(self, elem1: UIElement, elem2: UIElement, threshold: float = 0.5) -> bool:
        """Check if two UI elements overlap significantly."""
        x1, y1, w1, h1 = elem1.bbox
        x2, y2, w2, h2 = elem2.bbox
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left < right and top < bottom:
            intersection_area = (right - left) * (bottom - top)
            area1 = w1 * h1
            area2 = w2 * h2
            
            # Calculate overlap ratio
            overlap_ratio = intersection_area / min(area1, area2)
            return overlap_ratio > threshold
        
        return False
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False) 