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
# Import the vision utility
from qwen_vl_utils import process_vision_info

# Increase the PIL image size limit
Image.MAX_IMAGE_PIXELS = 200000000


class QwenVLProcessor:
    """Process PDF documents using Qwen2.5-VL-3B-Instruct for text recognition."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", use_cache: bool = True):
        """Initialize the Qwen VL processor with GPU support."""
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
        """Resize image if it exceeds memory-safe dimensions."""
        debug_dir = Path("/mnt/rectangularfile/debug_images")
        debug_dir.mkdir(exist_ok=True)
        
        # Save original image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_path = debug_dir / f"original_{timestamp}.jpg"
        image.save(original_path)
        self.logger.info(f"Saved original image to {original_path}")
        
        original_pixels = image.width * image.height
        if original_pixels <= self.target_image_pixels:
            return image

        ratio = (self.target_image_pixels / original_pixels) ** 0.5
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)

        self.logger.info(
            f"Resizing image from {image.width}x{image.height} "
            f"({original_pixels:,} pixels) to {new_width}x{new_height} "
            f"({new_width * new_height:,} pixels) to conserve memory"
        )

        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Save resized image
        resized_path = debug_dir / f"resized_{timestamp}.jpg"
        resized_image.save(resized_path)
        self.logger.info(f"Saved resized image to {resized_path}")
        
        return resized_image

    def _load_model(self):
        """Load the model, tokenizer, and processor if not already loaded."""
        if self.model is None:
            try:
                self.logger.info(f"Loading model: {self.model_name}")
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

                # Ensure tokenizer has pad and eos tokens if needed
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Optimize model loading based on device
                if self.device == "cuda":
                    self.logger.info("Loading model with INT8 quantization")
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                    model_kwargs = {
                        "device_map": "auto",
                        "trust_remote_code": True,
                        "quantization_config": quantization_config
                    }
                else:
                    self.logger.info("Loading model for CPU inference")
                    model_kwargs = {
                        "device_map": "auto",
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True
                    }

                # Load the model
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    **model_kwargs
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
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return False
        return True

    def process_document(
        self,
        pdf_path: Path,
        doc_id: int,
        db_manager: DatabaseManager,
        dpi: int = 150
    ) -> bool:
        """Process a PDF document for text recognition using Qwen2.5-VL."""
        try:
            self.logger.info(f"=== Starting Qwen VL processing for document {doc_id} ===")
            self.logger.info(f"Processing file: {pdf_path}")

            if not self._load_model():
                self.logger.error("Failed to load model")
                db_manager.update_processing_progress(doc_id, 0.0, "Failed to load model")
                return False

            if not db_manager.mark_ocr_started(doc_id):
                self.logger.error(f"Failed to mark document {doc_id} for processing")
                return False

            self.logger.info(f"Converting PDF to images at {dpi} DPI")

            with tempfile.TemporaryDirectory() as temp_dir:
                db_manager.update_processing_progress(doc_id, 10.0, "Converting PDF to images")

                try:
                    images = convert_from_path(
                        pdf_path,
                        dpi=dpi,
                        output_folder=temp_dir,
                        fmt='jpg'
                    )
                    self.logger.info(f"Converted PDF to {len(images)} images")
                    processed_images = images

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

                    text, confidence = self._process_image(image)

                    word_count = len(text.split()) if text else 0
                    char_count = len(text) if text else 0

                    text_sample = text[:100].replace('\n', ' ') if text else "[No text recognized]"
                    self.logger.info(f"Text sample: {text_sample}...")
                    self.logger.info(f"Recognized {word_count} words with confidence {confidence:.2f}")

                    page_data[page_num] = {
                        'text': text,
                        'confidence': confidence,
                        'word_count': word_count,
                        'char_count': char_count,
                        'processed_at': datetime.now()
                    }

                    if self.device == "cuda":
                        torch.cuda.empty_cache()

                self.logger.info(f"Storing recognition results for {len(page_data)} pages")
                db_manager.update_processing_progress(doc_id, 90.0, "Storing recognized text")

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
        """Process a single image with Qwen2.5-VL using the approach from model_test.py."""
        try:
            self.logger.info(f"Processing image of size {image.width}x{image.height}")
            self._log_memory_usage("Before image processing")

            image = self._resize_image_if_needed(image)
            
            # Create message structure matching model_test.py approach
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "resized_height": image.height,
                            "resized_width": image.width,
                        },
                        {"type": "text", "text": "Transcribe the handwritten text in this image as exactly as possible."},
                    ],
                }
            ]
            
            # Use apply_chat_template to format the text properly
            self.logger.info("Applying chat template...")
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process vision information with the utility function
            self.logger.info("Processing vision information...")
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Prepare inputs with the same approach as model_test.py
            self.logger.info("Preparing inputs with processor...")
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            self.logger.info("Processor output details:")
            for key, value in inputs.items():
                if hasattr(value, 'shape'):
                    self.logger.info(f"- {key}: shape {value.shape}")
                elif isinstance(value, list):
                    self.logger.info(f"- {key}: list of length {len(value)}")
                else:
                    self.logger.info(f"- {key}: type {type(value)}")

            inputs = inputs.to(self.device)
            self.logger.info(f"Moved inputs to {self.device}")
            self._log_memory_usage("After moving inputs to GPU")

            self.logger.info("Generating transcription...")
            with torch.no_grad():
                # Extract input_ids for trimming output later
                input_ids = inputs.input_ids
                
                # Generate output
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Trim input tokens from output, like in model_test.py
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
                ]
                
                # Decode output
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )
                
                # Use the first (and only) result
                text = output_text[0]

            self._log_memory_usage("After generation")
            self.logger.info("Generation completed")

            del inputs, generated_ids, generated_ids_trimmed
            if self.device == "cuda":
                torch.cuda.empty_cache()
                self._log_memory_usage("After cleanup")

            text_preview = text[:100] + "..." if text else "[No text recognized]"
            self.logger.info("=== Transcription preview ===")
            self.logger.info(text_preview)
            self.logger.info("============================")

            return text, 0.95 if text else 0.0

        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return "", 0.0