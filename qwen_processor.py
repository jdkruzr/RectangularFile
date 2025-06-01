import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import tempfile
import time

import torch
from transformers import AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from PIL import Image
from pdf2image import convert_from_path
from db_manager import DatabaseManager

# Increase the PIL image size limit
Image.MAX_IMAGE_PIXELS = 200000000

class QwenVLProcessor:
    """Process PDF documents using Qwen2-VL for text recognition with GPU acceleration."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B", use_cache: bool = True):
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
        
        # Check for GPU availability and set device
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.logger.info(f"Using GPU: {gpu_name} with {gpu_memory:.2f} GB VRAM")
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
    
    def _log_memory_usage(self):
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            self.logger.info(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    def _load_model(self):
        """Load the model, tokenizer, and processor if not already loaded."""
        if self.model is None:
            try:
                self.logger.info(f"Loading Qwen2-VL model: {self.model_name}")
                start_time = time.time()
                
                # Load tokenizer and processor
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True
                )
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                # Optimize model loading based on device
                if self.device == "cuda":
                    # Use 8-bit quantization for all models on GPU
                    self.logger.info("Loading model with INT8 quantization")
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        trust_remote_code=True,
                        quantization_config=quantization_config
                    )
                else:
                    # For CPU, use default settings
                    self.logger.info("Loading model for CPU inference")
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                
                # Put model in evaluation mode
                self.model.eval()
                
                # Log memory usage
                self._log_memory_usage()
                
                elapsed = time.time() - start_time
                self.logger.info(f"Model loaded successfully in {elapsed:.2f} seconds")
                return True
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                return False
        return True
    
    def process_document(
        self,
        pdf_path: Path,
        doc_id: int,
        db_manager: DatabaseManager,
        dpi: int = 150  # Good balance for GPU processing
    ) -> bool:
        """
        Process a PDF document for text recognition using Qwen2-VL with GPU acceleration.
        
        Args:
            pdf_path: Path to the PDF file
            doc_id: Database ID of the document
            db_manager: Database manager instance
            dpi: DPI for PDF to image conversion
            
        Returns:
            bool: True if processing was successful
        """
        try:
            self.logger.info(f"=== Starting Qwen VL processing for document {doc_id} ===")
            self.logger.info(f"Processing file: {pdf_path}")
            
            # Load model if not already loaded
            if not self._load_model():
                self.logger.error("Failed to load Qwen2-VL model")
                db_manager.update_processing_progress(
                    doc_id, 0.0, "Failed to load Qwen2-VL model"
                )
                return False
            
            # Mark document as being processed
            if not db_manager.mark_ocr_started(doc_id):
                self.logger.error(f"Failed to mark document {doc_id} for processing")
                return False
            
            # Convert PDF to images
            self.logger.info(f"Converting PDF to images at {dpi} DPI")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                db_manager.update_processing_progress(
                    doc_id, 10.0, "Converting PDF to images"
                )

                try:
                    images = convert_from_path(
                        pdf_path,
                        dpi=dpi,
                        output_folder=temp_dir,
                        fmt='jpg'
                    )
                    self.logger.info(f"Converted PDF to {len(images)} images")
                    
                    # Process images - resize if needed
                    processed_images = []
                    
                    # With GPU, we can handle larger images
                    max_pixels = 3000000 if self.device == "cuda" else 1000000
                    
                    for img in images:
                        # Check if image is too large
                        if img.width * img.height > max_pixels:
                            # Calculate new dimensions keeping aspect ratio
                            scale_factor = (max_pixels / (img.width * img.height)) ** 0.5
                            new_width = int(img.width * scale_factor)
                            new_height = int(img.height * scale_factor)
                            
                            # Resize image
                            self.logger.info(f"Resizing image from {img.width}x{img.height} to {new_width}x{new_height}")
                            img = img.resize((new_width, new_height), Image.LANCZOS)
                        
                        processed_images.append(img)
                    
                    self.logger.info(f"Prepared {len(processed_images)} images for processing")
                    
                except Exception as pdf_error:
                    self.logger.error(f"Error converting PDF to images: {pdf_error}")
                    db_manager.update_processing_progress(
                        doc_id, 0.0, f"Error converting PDF to images: {str(pdf_error)}"
                    )
                    return False

                if not processed_images:
                    self.logger.warning(f"No images extracted from PDF {doc_id}")
                    db_manager.update_processing_progress(
                        doc_id, 0.0, "No images could be extracted from PDF"
                    )
                    return False

                # Process each page
                page_data = {}
                progress_per_page = 80.0 / len(processed_images)

                for i, image in enumerate(processed_images):
                    page_num = i + 1
                    current_progress = 20.0 + (i * progress_per_page)

                    self.logger.info(f"Processing page {page_num}/{len(processed_images)}")
                    db_manager.update_processing_progress(
                        doc_id,
                        current_progress,
                        f"Recognizing text on page {page_num}/{len(processed_images)}"
                    )

                    # Process the image with Qwen2-VL
                    text, confidence = self._process_image(image)
                    
                    # Count words and characters
                    word_count = len(text.split()) if text else 0
                    char_count = len(text) if text else 0

                    # Log sample of recognized text
                    text_sample = text[:100].replace('\n', ' ') if text else "[No text recognized]"
                    self.logger.info(f"Text sample: {text_sample}...")
                    self.logger.info(f"Recognized {word_count} words with confidence {confidence:.2f}")
                    
                    # Store the OCR result
                    page_data[page_num] = {
                        'text': text,
                        'confidence': confidence,
                        'word_count': word_count,
                        'char_count': char_count,
                        'processed_at': datetime.now()
                    }
                    
                    # Clear GPU cache between pages if using CUDA
                    if self.device == "cuda":
                        torch.cuda.empty_cache()

                # Store results in database
                self.logger.info(f"Storing recognition results for {len(page_data)} pages")
                db_manager.update_processing_progress(
                    doc_id, 90.0, "Storing recognized text"
                )
            
                success = db_manager.store_ocr_results(doc_id, page_data)

                if success:
                    total_words = sum(page['word_count'] for page in page_data.values())
                    avg_conf = sum(page['confidence'] for page in page_data.values()) / len(page_data) if page_data else 0

                    self.logger.info(
                        f"Successfully processed document {doc_id}\n"
                        f"Total words: {total_words}\n"
                        f"Average confidence: {avg_conf:.2f}\n"
                        f"Pages processed: {len(page_data)}"
                    )
                else:
                    self.logger.error(f"Failed to store results for document {doc_id}")
                    db_manager.update_processing_progress(
                        doc_id, 0.0, "Failed to store recognition results"
                    )

                return success
        except Exception as e:
            error_message = f"Error processing document: {str(e)}"
            self.logger.error(f"Error in processing document {doc_id}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            db_manager.update_processing_progress(doc_id, 0.0, error_message)
            return False
    
    def _process_image(self, image: Image.Image) -> Tuple[str, float]:
        """
        Process a single image with Qwen2-VL using a sliding window approach.
        """
        try:
            # Window parameters - adjusted to match known working dimensions
            window_height = 1500  # Height of each window
            window_width = 1500   # Width of each window
            overlap = 150         # Overlap between windows (10% of window size)
            
            # Log original image size
            self.logger.info(f"Original image size: {image.width}x{image.height}")
            
            # If image is smaller than window size, process whole image
            if image.width <= window_width and image.height <= window_height:
                self.logger.info("Image smaller than window size, processing as single window")
                return self._process_single_window(image, "Please transcribe all handwritten text in this image, preserving layout:")
            
            # Calculate number of windows needed
            n_windows_y = max(1, (image.height - overlap) // (window_height - overlap))
            n_windows_x = max(1, (image.width - overlap) // (window_width - overlap))
            
            self.logger.info(f"Processing image with {n_windows_x}x{n_windows_y} windows")
            
            all_text_parts = []
            
            # Process each window
            for y in range(n_windows_y):
                for x in range(n_windows_x):
                    # Calculate window coordinates
                    x_start = x * (window_width - overlap)
                    y_start = y * (window_height - overlap)
                    x_end = min(x_start + window_width, image.width)
                    y_end = min(y_start + window_height, image.height)
                    
                    # Extract window
                    window = image.crop((x_start, y_start, x_end, y_end))
                    
                    # Resize window to match model's expected dimensions
                    aspect_ratio = window.width / window.height
                    if aspect_ratio > 1:
                        new_width = 1500
                        new_height = int(1500 / aspect_ratio)
                    else:
                        new_height = 1500
                        new_width = int(1500 * aspect_ratio)
                    
                    window = window.resize((new_width, new_height), Image.LANCZOS)
                    
                    self.logger.info(f"Processing window at ({x_start},{y_start}) to ({x_end},{y_end}), resized to {new_width}x{new_height}")
                    
                    # Process window
                    prompt = f"Please transcribe any handwritten text in this section of the image, preserving layout. This is part {x+1},{y+1} of {n_windows_x*n_windows_y}:"
                    
                    text, _ = self._process_single_window(window, prompt)
                    
                    if text:
                        all_text_parts.append({
                            'text': text,
                            'position': (x_start, y_start, x_end, y_end)
                        })
                    
                    # Clear GPU cache between windows
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
            
            # Combine text parts
            if all_text_parts:
                all_text_parts.sort(key=lambda p: (p['position'][1], p['position'][0]))
                combined_text = ""
                last_y = -1
                
                for part in all_text_parts:
                    text = part['text']
                    y_start = part['position'][1]
                    
                    if last_y != -1 and abs(y_start - last_y) > overlap:
                        combined_text += "\n\n"
                    elif last_y != -1:
                        combined_text += " "
                    
                    combined_text += text
                    last_y = y_start
                
                self.logger.info(f"Successfully combined text from {len(all_text_parts)} windows")
                return combined_text.strip(), 0.95
            
            return "", 0.0
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return "", 0.0

    def _process_single_window(self, image: Image.Image, prompt: str) -> Tuple[str, float]:
        """Helper method to process a single window of the image."""
        try:
            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=False,
                    temperature=1.0
                )
            
            # Decode the output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up
            del inputs, outputs
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Extract text from response
            if prompt in generated_text:
                text = generated_text.replace(prompt, "").strip()
            else:
                parts = generated_text.split(":")
                text = parts[-1].strip() if len(parts) > 1 else generated_text.strip()
            
            return text, 0.95
            
        except Exception as e:
            self.logger.error(f"Error processing window: {e}")
            return "", 0.0