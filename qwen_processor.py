import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import tempfile
import time

# Set CUDA memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from transformers import (
    AutoTokenizer, 
    AutoProcessor, 
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration
)
from PIL import Image
from pdf2image import convert_from_path
from db_manager import DatabaseManager

# Increase the PIL image size limit
Image.MAX_IMAGE_PIXELS = 200000000

class QwenVLProcessor:
    """Process PDF documents using Qwen2.5-VL-3B-Instruct for text recognition."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", use_cache: bool = True):
        """
        Initialize the Qwen VL processor with GPU support.
        
        Args:
            model_name: Name or path of the Qwen VL model
            use_cache: Whether to use the Hugging Face cache
        """
        # Set cache directory
        os.environ['TRANSFORMERS_CACHE'] = '/mnt/rectangularfile/qwencache'
        
        self.setup_logging()
        self.model_name = model_name
        self.use_cache = use_cache
        
        # Memory management settings
        self.max_image_dimension = 1500  # Max dimension for input images
        self.target_image_pixels = 1500 * 1500  # Target total pixels
        
        # Check for GPU availability and set device
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.logger.info(f"Using GPU: {gpu_name} with {gpu_memory:.2f} GB VRAM")
            
            # Log initial GPU memory state
            self._log_memory_usage(note="Initial GPU memory state")
        else:
            self.device = "cpu"
            self.logger.info("No GPU detected, using CPU")
        
        # Lazy loading - will initialize when first used
        self.tokenizer = None
        self.processor = None
        self.model = None
    
    def setup_logging(self):
        """Configure logging for the processor."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _log_memory_usage(self, note: str = ""):
        """Log current GPU memory usage with optional note."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            free = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) - allocated
            
            memory_info = (
                f"GPU Memory: {allocated:.2f} GB allocated, "
                f"{reserved:.2f} GB reserved, "
                f"{free:.2f} GB free"
            )
            if note:
                memory_info = f"{note}: {memory_info}"
            
            self.logger.info(memory_info)
    
    def _resize_image_if_needed(self, image: Image.Image) -> Image.Image:
        """
        Resize image if it exceeds memory-safe dimensions.
        
        The 3B model requires more memory than the 2B version, so we need to be
        more aggressive with image resizing to prevent OOM errors.
        """
        original_pixels = image.width * image.height
        if original_pixels <= self.target_image_pixels:
            return image
            
        # Calculate new dimensions maintaining aspect ratio
        ratio = (self.target_image_pixels / original_pixels) ** 0.5
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)
        
        self.logger.info(
            f"Resizing image from {image.width}x{image.height} "
            f"({original_pixels:,} pixels) to {new_width}x{new_height} "
            f"({new_width * new_height:,} pixels) to conserve memory"
        )
        
        return image.resize((new_width, new_height), Image.LANCZOS)

    def _process_image(self, image: Image.Image) -> Tuple[str, float]:
        """Process a single image with Qwen2.5-VL."""
        try:
            self.logger.info(f"Processing image of size {image.width}x{image.height}")
            self._log_memory_usage("Before image processing")
            
            # Resize image if needed to prevent OOM
            image = self._resize_image_if_needed(image)
            
            # Qwen2.5-VL-Instruct specific prompt format
            prompt = """<|im_start|>system
You are a helpful assistant that accurately transcribes handwritten text from images.
<|im_end|>
<|im_start|>user
Please transcribe all handwritten text in this image, preserving layout and line breaks.
<|im_end|>
<|im_start|>assistant
I'll transcribe the handwritten text from the image, maintaining its layout:
"""
            
            # Process with explicit processor settings and logging
            self.logger.info("Preparing inputs with processor...")
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Log processor output details
            self.logger.info("Processor output details:")
            for key, value in inputs.items():
                if hasattr(value, 'shape'):
                    self.logger.info(f"- {key}: shape {value.shape}")
                elif isinstance(value, list):
                    self.logger.info(f"- {key}: list of length {len(value)}")
                else:
                    self.logger.info(f"- {key}: type {type(value)}")
            
            # Move to device and log memory
            inputs = inputs.to(self.device)
            self.logger.info(f"Moved inputs to {self.device}")
            self._log_memory_usage("After moving inputs to GPU")
            
            # Generate transcription
            self.logger.info("Generating transcription...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            self._log_memory_usage("After generation")
            self.logger.info("Generation completed")
            
            # Decode the output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Clean up and force memory release
            del inputs, outputs
            if self.device == "cuda":
                torch.cuda.empty_cache()
                self._log_memory_usage("After cleanup")
            
            # Extract text between assistant's response and end token
            response_start = generated_text.find("<|im_start|>assistant")
            if response_start != -1:
                text_start = generated_text.find("\n", response_start) + 1
                text_end = generated_text.find("<|im_end|>", text_start)
                if text_end == -1:
                    text_end = None
                text = generated_text[text_start:text_end].strip()
            else:
                text = generated_text.strip()
            
            # Log result summary
            text_preview = text[:100] + "..." if text else "[No text recognized]"
            self.logger.info(f"Recognized text preview: {text_preview}")
            
            return text, 0.95 if text else 0.0
                
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return "", 0.0