import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import tempfile
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from PIL import Image
from pdf2image import convert_from_path
from db_manager import DatabaseManager

# Increase the PIL image size limit to prevent DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = 200000000  # Increased limit for large images

class QwenVLProcessor:
    """Process PDF documents using Qwen2-VL-7B for text recognition."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B", use_cache: bool = True):
        """
        Initialize the Qwen VL processor.
        
        Args:
            model_name: Name or path of the Qwen VL model
            use_cache: Whether to use the Hugging Face cache
        """
        self.setup_logging()
        self.model_name = model_name
        self.use_cache = use_cache
        
        # Determine device (GPU if available, else CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
        
        # Lazy loading - will initialize when first used
        self.tokenizer = None
        self.processor = None
        self.model = None

        # Set custom cache directory
        os.environ['TRANSFORMERS_CACHE'] = '/mnt/rectangularfile/qwencache'
    
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
                
                # Check if we can use quantization
                can_use_quantization = False
                try:
                    import bitsandbytes
                    if torch.cuda.is_available():
                        can_use_quantization = True
                except (ImportError, AttributeError):
                    self.logger.warning("bitsandbytes not available for quantization, using full precision")
                
                # Load model with appropriate configuration
                if can_use_quantization:
                    self.logger.info("Using 4-bit quantization with CUDA")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4"
                    )
                    
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        quantization_config=quantization_config
                    )
                else:
                    self.logger.info("Using CPU without quantization")
                    # For CPU-only systems, don't use quantization
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float32,  # Use float32 for CPU
                        offload_folder="/mnt/rectangularfile/qwencache/offload"  # Helps with memory management
                    )
                
                # Put model in evaluation mode
                self.model.eval()
                
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
        dpi: int = 150  # Changed from 300 to 150 for better performance
    ) -> bool:
        """
        Process a PDF document for text recognition using Qwen2-VL.
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
                    
                    # Process images - resize large ones
                    processed_images = []
                    for img in images:
                        # Check if image is too large
                        if img.width * img.height > 4000000:  # 4 million pixels
                            # Calculate new dimensions keeping aspect ratio
                            scale_factor = (4000000 / (img.width * img.height)) ** 0.5
                            new_width = int(img.width * scale_factor)
                            new_height = int(img.height * scale_factor)
                            
                            # Resize image
                            self.logger.info(f"Resizing large image from {img.width}x{img.height} to {new_width}x{new_height}")
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

                    # Add explicit cleanup after each page to manage memory
                    if i < len(processed_images) - 1:  # If not the last page
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        import gc
                        gc.collect()  # Force garbage collection between pages

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
        Process a single image with Qwen2-VL.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Tuple of (recognized_text, confidence_score)
        """
        try:
            # Define prompt for handwriting recognition
            prompt = "Please transcribe all the handwritten text in this image, preserving the layout and line breaks:"
            
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
                    max_new_tokens=800,  # Allow for longer outputs
                    do_sample=False,     # Deterministic generation
                    temperature=1.0      # Default temperature
                )
            
            # Decode the output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the output
            if prompt in generated_text:
                transcription = generated_text.replace(prompt, "").strip()
            else:
                # If prompt not found in output, take everything after a reasonable marker
                parts = generated_text.split(":")
                if len(parts) > 1:
                    transcription = parts[1].strip()
                else:
                    transcription = generated_text.strip()
            
            # For confidence, since the model doesn't provide token-level confidences,
            # we use a placeholder high value (these models are generally quite accurate)
            confidence = 0.95
            
            return transcription, confidence
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return "", 0.0