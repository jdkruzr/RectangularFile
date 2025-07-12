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
        self.yellow_lower = np.array([15, 40, 40])   # Broader HSV range for yellow
        self.yellow_upper = np.array([35, 255, 255]) # Upper HSV bound for yellow
        
        # Green box detection parameters (HSV color space)
        self.green_lower = np.array([40, 50, 50])    # Lower HSV bound for green
        self.green_upper = np.array([80, 255, 255])  # Upper HSV bound for green
        
        # Minimum area threshold for regions (to filter noise)
        self.min_area = 5000  # Balanced threshold for phrases and list items
        
        # Morphological operations kernel sizes
        self.highlight_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # Larger kernel to merge nearby regions
        self.box_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # Minimum dimensions for valid regions
        self.min_width = 150   # Reduced for individual list items
        self.min_height = 25   # Minimum height in pixels
    
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
            
            # Moderate dilation to merge words into phrases (dialed back from 100)
            # Use wide horizontal kernel to bridge word gaps
            mega_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 12))  # Less aggressive
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_DILATE, mega_dilate_kernel)
            
            # Also try vertical dilation for list-style highlighting
            vertical_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 60))  # Smaller for lists
            yellow_mask_vertical = cv2.morphologyEx(yellow_mask, cv2.MORPH_DILATE, vertical_dilate_kernel)
            
            # Combine horizontal and vertical dilations
            yellow_mask = cv2.bitwise_or(yellow_mask, yellow_mask_vertical)
            
            # Multiple rounds of horizontal closing to connect distant words
            close_kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 8))
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, close_kernel1)
            close_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 5))
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, close_kernel2)
            
            # Find contours
            contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            logger.info(f"Found {len(contours)} yellow contours before filtering")
            
            highlights = []
            filtered_count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    filtered_count += 1
                    logger.debug(f"Filtered yellow contour: area {area} < {self.min_area}")
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by minimum dimensions
                if w < self.min_width or h < self.min_height:
                    filtered_count += 1
                    logger.debug(f"Filtered yellow contour: dimensions {w}x{h} < {self.min_width}x{self.min_height}")
                    continue
                
                highlights.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'type': 'yellow_highlight'
                })
            
            # Sort by area (largest first)
            highlights.sort(key=lambda x: x['area'], reverse=True)
            
            logger.info(f"Detected {len(highlights)} yellow highlight regions (filtered out {filtered_count})")
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
                
                boxes.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'type': 'green_box'
                })
            
            # Sort by area (largest first)
            boxes.sort(key=lambda x: x['area'], reverse=True)
            
            logger.info(f"Detected {len(boxes)} green box regions")
            return boxes
            
        except Exception as e:
            logger.error(f"Error detecting green boxes: {e}")
            return []
    
    
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