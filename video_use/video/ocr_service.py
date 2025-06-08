"""OCR service for extracting text from video frames."""

import cv2
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False
    logging.warning("pytesseract not available. Install it for OCR functionality.")

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False
    logging.warning("easyocr not available. Install it for enhanced OCR functionality.")

from ..schema.models import Frame, VideoAnalysisConfig

logger = logging.getLogger(__name__)


class OCRResult:
    """Represents OCR result for a text region."""
    
    def __init__(self, text: str, bbox: Tuple[int, int, int, int], confidence: float):
        self.text = text.strip()
        self.bbox = bbox  # (x, y, width, height)
        self.confidence = confidence
    
    def __repr__(self):
        return f"OCRResult(text='{self.text}', confidence={self.confidence:.2f})"


class OCRService:
    """Service for extracting text from video frames using multiple OCR engines."""
    
    def __init__(self, config: VideoAnalysisConfig):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.easyocr_reader = None
        
        # Initialize EasyOCR if available
        if HAS_EASYOCR and self.config.enable_ocr:
            try:
                self.easyocr_reader = easyocr.Reader(self.config.ocr_languages)
                logger.info(f"EasyOCR initialized with languages: {self.config.ocr_languages}")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
    
    async def extract_text_from_frame(self, frame: Frame) -> List[OCRResult]:
        """Extract text from a single frame using available OCR engines."""
        if not self.config.enable_ocr:
            return []
        
        try:
            loop = asyncio.get_event_loop()
            
            # Try EasyOCR first (usually more accurate)
            if self.easyocr_reader:
                results = await loop.run_in_executor(
                    self.executor,
                    self._extract_with_easyocr,
                    frame.image
                )
                if results:
                    logger.debug(f"EasyOCR found {len(results)} text regions in frame {frame.frame_number}")
                    return results
            
            # Fallback to Tesseract
            if HAS_PYTESSERACT:
                results = await loop.run_in_executor(
                    self.executor,
                    self._extract_with_tesseract,
                    frame.image
                )
                logger.debug(f"Tesseract found {len(results)} text regions in frame {frame.frame_number}")
                return results
            
            logger.warning("No OCR engines available")
            return []
            
        except Exception as e:
            logger.error(f"Error extracting text from frame {frame.frame_number}: {e}")
            return []
    
    def _extract_with_easyocr(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text using EasyOCR."""
        try:
            results = self.easyocr_reader.readtext(image)
            ocr_results = []
            
            for bbox, text, confidence in results:
                if confidence > 0.5 and text.strip():  # Filter low confidence results
                    # Convert bbox format from EasyOCR to our format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    x = int(min(x_coords))
                    y = int(min(y_coords))
                    width = int(max(x_coords) - min(x_coords))
                    height = int(max(y_coords) - min(y_coords))
                    
                    ocr_results.append(OCRResult(
                        text=text,
                        bbox=(x, y, width, height),
                        confidence=confidence
                    ))
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return []
    
    def _extract_with_tesseract(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text using Tesseract OCR."""
        try:
            # Configure Tesseract
            config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?@#$%^&*()_+-=[]{}|;:,.<>?'
            
            # Get bounding box data
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            
            ocr_results = []
            n_boxes = len(data['level'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                confidence = float(data['conf'][i])
                
                # Filter out low confidence and empty results
                if confidence > 50 and text:
                    x = data['left'][i]
                    y = data['top'][i]
                    width = data['width'][i]
                    height = data['height'][i]
                    
                    ocr_results.append(OCRResult(
                        text=text,
                        bbox=(x, y, width, height),
                        confidence=confidence / 100.0  # Normalize to 0-1
                    ))
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return []
    
    async def extract_text_from_frames(self, frames: List[Frame]) -> Dict[int, List[OCRResult]]:
        """Extract text from multiple frames in parallel."""
        if not self.config.enable_ocr or not frames:
            return {}
        
        logger.info(f"Extracting text from {len(frames)} frames")
        
        # Process frames in parallel
        tasks = []
        for frame in frames:
            task = self.extract_text_from_frame(frame)
            tasks.append(task)
        
        # Wait for all OCR tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compile results by frame number
        frame_texts = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"OCR failed for frame {frames[i].frame_number}: {result}")
                frame_texts[frames[i].frame_number] = []
            else:
                frame_texts[frames[i].frame_number] = result
        
        # Log summary
        total_texts = sum(len(texts) for texts in frame_texts.values())
        logger.info(f"OCR completed: {total_texts} text regions found across {len(frames)} frames")
        
        return frame_texts
    
    async def detect_text_changes(self, frames: List[Frame]) -> List[Tuple[int, int, str]]:
        """Detect when text content changes between frames."""
        frame_texts = await self.extract_text_from_frames(frames)
        text_changes = []
        
        prev_texts = set()
        
        for frame in frames:
            current_texts = set(
                result.text for result in frame_texts.get(frame.frame_number, [])
                if result.confidence > 0.7
            )
            
            # Find new text that appeared
            new_texts = current_texts - prev_texts
            
            # Find text that disappeared  
            removed_texts = prev_texts - current_texts
            
            if new_texts or removed_texts:
                change_description = []
                if new_texts:
                    change_description.append(f"Added: {', '.join(new_texts)}")
                if removed_texts:
                    change_description.append(f"Removed: {', '.join(removed_texts)}")
                
                text_changes.append((
                    frame.frame_number,
                    frame.frame_number - (frames[0].frame_number if frames else 0),
                    "; ".join(change_description)
                ))
            
            prev_texts = current_texts
        
        return text_changes
    
    async def find_ui_text_elements(self, frame: Frame) -> List[Dict[str, Any]]:
        """Find text elements that likely represent UI components."""
        ocr_results = await self.extract_text_from_frame(frame)
        
        ui_elements = []
        
        for result in ocr_results:
            if result.confidence < 0.6:
                continue
            
            # Classify text as potential UI element
            ui_type = self._classify_ui_text(result.text)
            
            if ui_type:
                ui_elements.append({
                    'type': ui_type,
                    'text': result.text,
                    'bbox': result.bbox,
                    'confidence': result.confidence,
                    'frame_number': frame.frame_number
                })
        
        return ui_elements
    
    def _classify_ui_text(self, text: str) -> Optional[str]:
        """Classify text as a UI element type."""
        text_lower = text.lower().strip()
        
        # Common button text patterns
        button_patterns = [
            'click', 'submit', 'send', 'save', 'cancel', 'ok', 'yes', 'no',
            'login', 'sign in', 'sign up', 'register', 'continue', 'next',
            'back', 'previous', 'finish', 'done', 'apply', 'confirm'
        ]
        
        # Form field labels
        label_patterns = [
            'name', 'email', 'password', 'username', 'phone', 'address',
            'city', 'state', 'zip', 'country', 'first name', 'last name'
        ]
        
        # Navigation elements
        nav_patterns = [
            'home', 'about', 'contact', 'services', 'products', 'help',
            'settings', 'profile', 'dashboard', 'menu'
        ]
        
        if any(pattern in text_lower for pattern in button_patterns):
            return 'button'
        elif any(pattern in text_lower for pattern in label_patterns):
            return 'label'
        elif any(pattern in text_lower for pattern in nav_patterns):
            return 'navigation'
        elif text.endswith(':') or text.endswith('*'):
            return 'label'
        elif len(text.split()) == 1 and text.isalpha():
            return 'button'
        
        return None
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False) 