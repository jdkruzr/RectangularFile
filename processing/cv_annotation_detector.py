import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class CVAnnotationDetector:
    """Computer vision-based annotation detection for handwritten documents"""
    
    def __init__(self):
        # Yellow highlight detection parameters (HSV color space)
        self.yellow_lower = np.array([20, 50, 50])   # Lower HSV bound for yellow
        self.yellow_upper = np.array([30, 255, 255]) # Upper HSV bound for yellow
        
        # Green box detection parameters (HSV color space)
        self.green_lower = np.array([40, 50, 50])    # Lower HSV bound for green
        self.green_upper = np.array([80, 255, 255])  # Upper HSV bound for green
        
        # Minimum area threshold for regions (to filter noise)
        self.min_area = 8000  # Very high threshold to ensure only full phrases
        
        # Morphological operations kernel sizes
        self.highlight_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # Larger kernel to merge nearby regions
        self.box_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # Minimum dimensions for valid regions
        self.min_width = 300   # Minimum width in pixels - must be phrase-sized
        self.min_height = 30   # Minimum height in pixels
    
    def detect_yellow_highlights(self, image: Image.Image) -> List[Dict]:
        """
        Detect yellow highlighted regions in the image
        
        Args:
            image: PIL Image object
            
        Returns:
            List of dictionaries with bounding boxes and confidence scores
        """
        try:
            # Convert PIL image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Create mask for yellow regions
            yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
            
            # Clean up the mask with morphological operations
            # First close small gaps to connect nearby regions
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, self.highlight_kernel)
            # Then open to remove small noise
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, self.highlight_kernel)
            
            # VERY aggressive dilation to merge words into phrases
            # Use extremely wide horizontal kernel to bridge word gaps
            mega_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 15))  # Very wide
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_DILATE, mega_dilate_kernel)
            
            # Multiple rounds of horizontal closing to connect distant words
            close_kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 8))
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, close_kernel1)
            close_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 5))
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, close_kernel2)
            
            # Find contours
            contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            highlights = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by minimum dimensions
                if w < self.min_width or h < self.min_height:
                    continue
                
                # Calculate confidence based on area and shape
                confidence = self._calculate_highlight_confidence(area, w, h, yellow_mask, x, y)
                
                highlights.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'confidence': confidence,
                    'type': 'yellow_highlight'
                })
            
            # Sort by confidence (highest first)
            highlights.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.info(f"Detected {len(highlights)} yellow highlight regions")
            return highlights
            
        except Exception as e:
            logger.error(f"Error detecting yellow highlights: {e}")
            return []
    
    def detect_green_boxes(self, image: Image.Image) -> List[Dict]:
        """
        Detect green boxed regions in the image
        
        Args:
            image: PIL Image object
            
        Returns:
            List of dictionaries with bounding boxes and confidence scores
        """
        try:
            # Convert PIL image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Create mask for green regions
            green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            
            # Clean up the mask
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, self.box_kernel)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, self.box_kernel)
            
            # Find contours
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            boxes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by minimum dimensions
                if w < self.min_width or h < self.min_height:
                    continue
                
                # For boxes, we want to check if it's actually a rectangular outline
                # rather than a filled region
                confidence = self._calculate_box_confidence(contour, area, w, h, green_mask, x, y)
                
                boxes.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'confidence': confidence,
                    'type': 'green_box'
                })
            
            # Sort by confidence (highest first)
            boxes.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.info(f"Detected {len(boxes)} green box regions")
            return boxes
            
        except Exception as e:
            logger.error(f"Error detecting green boxes: {e}")
            return []
    
    def _calculate_highlight_confidence(self, area: int, width: int, height: int, 
                                      mask: np.ndarray, x: int, y: int) -> float:
        """Calculate confidence score for yellow highlight detection"""
        # Base confidence from area (larger = more confident)
        area_confidence = min(area / 1000, 1.0)
        
        # Aspect ratio confidence (highlights are usually wider than tall)
        aspect_ratio = width / height if height > 0 else 0
        aspect_confidence = min(aspect_ratio / 3.0, 1.0) if aspect_ratio > 1 else 0.5
        
        # Pixel density confidence (what percentage of bounding box is actually yellow)
        roi = mask[y:y+height, x:x+width]
        pixel_density = np.sum(roi > 0) / (width * height) if width * height > 0 else 0
        
        # Combine confidences
        confidence = (area_confidence * 0.4 + aspect_confidence * 0.3 + pixel_density * 0.3)
        return min(confidence, 1.0)
    
    def _calculate_box_confidence(self, contour: np.ndarray, area: int, width: int, height: int,
                                 mask: np.ndarray, x: int, y: int) -> float:
        """Calculate confidence score for green box detection"""
        # Check if contour is roughly rectangular
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # More vertices = more likely to be a box outline
        vertex_confidence = min(len(approx) / 4.0, 1.0)
        
        # Area confidence
        area_confidence = min(area / 1000, 1.0)
        
        # For boxes, we want hollow rectangles, so lower pixel density is better
        roi = mask[y:y+height, x:x+width]
        pixel_density = np.sum(roi > 0) / (width * height) if width * height > 0 else 0
        
        # Inverse pixel density (hollow boxes have lower density)
        hollow_confidence = 1.0 - min(pixel_density, 1.0)
        
        # Combine confidences
        confidence = (vertex_confidence * 0.4 + area_confidence * 0.3 + hollow_confidence * 0.3)
        return min(confidence, 1.0)
    
    def extract_region_image(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        """
        Extract a region from the image based on bounding box
        
        Args:
            image: PIL Image object
            bbox: (x, y, width, height) tuple
            
        Returns:
            PIL Image of the extracted region
        """
        x, y, w, h = bbox
        return image.crop((x, y, x + w, y + h))
    
    def detect_all_annotations(self, image: Image.Image) -> Dict[str, List[Dict]]:
        """
        Detect both yellow highlights and green boxes in one call
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with 'yellow_highlights' and 'green_boxes' keys
        """
        return {
            'yellow_highlights': self.detect_yellow_highlights(image),
            'green_boxes': self.detect_green_boxes(image)
        }