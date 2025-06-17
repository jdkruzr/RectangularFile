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
from db.db_manager import DatabaseManager
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

    def _ensure_annotations_table(self, db_manager):
        """Ensure the annotations table exists in the database."""
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS document_annotations (
                        id INTEGER PRIMARY KEY,
                        doc_id INTEGER,
                        page_number INTEGER,
                        annotation_type TEXT,  -- 'box', 'star', 'underline', etc.
                        text TEXT,
                        confidence FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (doc_id) REFERENCES pdf_documents(id)
                    )
                """)
                conn.commit()
                self.logger.info("Ensured document_annotations table exists")
        except Exception as e:
            self.logger.error(f"Error ensuring annotations table: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

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

    def _resize_image_if_needed(self, image: Image.Image, doc_id: int, page_num: int) -> Tuple[Image.Image, str]:
        """Resize image if it exceeds memory-safe dimensions and save it to disk."""
        debug_dir = Path("/mnt/rectangularfile/debug_images")
        debug_dir.mkdir(exist_ok=True)
        
        # Create a timestamp and filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_path = debug_dir / f"original_{timestamp}_doc_{doc_id}_page_{page_num}.jpg"
        image.save(original_path)
        self.logger.info(f"Saved original image to {original_path}")
        
        original_pixels = image.width * image.height
        if original_pixels <= self.target_image_pixels:
            return image, str(original_path)

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
        resized_path = debug_dir / f"resized_{timestamp}_doc_{doc_id}_page_{page_num}.jpg"
        resized_image.save(resized_path)
        self.logger.info(f"Saved resized image to {resized_path}")
        
        return resized_image, str(original_path)  # Return the original image path

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
                    self.logger.info("Loading model with INT8 quantization and memory efficiency")
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_skip_modules=None,
                        llm_int8_enable_fp32_cpu_offload=True
                    )
                    model_kwargs = {
                        "device_map": "auto",
                        "trust_remote_code": True,
                        "quantization_config": quantization_config,
                        "max_memory": {0: "10GiB"},  # Limit to 10GB VRAM usage
                        "offload_folder": "offload_folder"
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

    def process_document(self, pdf_path: Path, doc_id: int, db_manager: DatabaseManager, dpi: int = 150, detect_annotations: bool = True) -> bool:
        """
        Process a PDF document with a two-pass approach for text and annotations.
        
        Args:
            pdf_path: Path to the PDF file
            doc_id: Document ID in the database
            db_manager: Database manager instance
            dpi: DPI for image conversion
            detect_annotations: Whether to run the second pass for annotation detection
        
        Returns:
            bool: Success or failure
        """
        try:
            self.logger.info(f"=== Starting Qwen VL processing for document {doc_id} ===")
            self.logger.info(f"Processing file: {pdf_path}")
            self.logger.info(f"Annotation detection enabled: {detect_annotations}")


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

                # Convert PDF to images
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
                progress_per_page = 60.0 / len(processed_images)  # Reduced from 80 to allow for second pass
                annotations = []

                # First pass - Text transcription
                self.logger.info("Starting first pass: Text transcription")
                for i, image in enumerate(processed_images):
                    page_num = i + 1
                    current_progress = 20.0 + (i * progress_per_page)

                    self.logger.info(f"Transcribing page {page_num}/{len(processed_images)}")
                    db_manager.update_processing_progress(
                        doc_id,
                        current_progress,
                        f"Transcribing text on page {page_num}/{len(processed_images)}"
                    )

                    # Resize and save the image
                    image, image_path = self._resize_image_if_needed(image, doc_id, page_num)
                    
                    # Process the image for basic text transcription
                    text = self._transcribe_text(image)
                    
                    word_count = len(text.split()) if text else 0
                    char_count = len(text) if text else 0

                    page_data[page_num] = {
                        'text': text,
                        'word_count': word_count,
                        'char_count': char_count,
                        'processed_at': datetime.now(),
                        'image_path': image_path
                    }

                    if self.device == "cuda":
                        torch.cuda.empty_cache()

                # Store the transcription results
                self.logger.info(f"Storing transcription results for {len(page_data)} pages")
                db_manager.update_processing_progress(doc_id, 80.0, "Storing transcribed text")
                success = db_manager.store_ocr_results(doc_id, page_data)

                if not success:
                    self.logger.error(f"Failed to store transcription results for document {doc_id}")
                    db_manager.update_processing_progress(doc_id, 0.0, "Failed to store transcription results")
                    return False

                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    self._log_memory_usage("After first pass cleanup")
                    
                    # If still low on memory, try to force Python garbage collection
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    self._log_memory_usage("After garbage collection")

                if not detect_annotations:
                    self.logger.info("Skipping annotation detection as requested")
                    db_manager.update_processing_progress(doc_id, 100.0, "Processing complete")
                    total_words = sum(page['word_count'] for page in page_data.values())
                    self.logger.info(
                        f"Successfully processed document {doc_id}\n"
                        f"Total words: {total_words}\n"
                        f"Pages processed: {len(page_data)}\n"
                        f"Annotations detection skipped"
                    )
                    return True

                # Second pass - Detect annotations
                self.logger.info("Starting second pass: Detecting annotations")
                db_manager.update_processing_progress(doc_id, 85.0, "Detecting annotations")
                
                # Ensure annotations table exists
                self._ensure_annotations_table(db_manager)
                
                for i, image in enumerate(processed_images):
                    page_num = i + 1
                    self.logger.info(f"Detecting annotations on page {page_num}/{len(processed_images)}")
                    
                    # Get boxed text annotations
                    page_annotations = self._detect_boxed_text(image, page_num)
                    annotations.extend(page_annotations)
                    
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                
                # Store annotations
                if annotations:
                    self.logger.info(f"Storing {len(annotations)} annotations")
                    db_manager.store_document_annotations(doc_id, annotations)
                
                db_manager.update_processing_progress(doc_id, 100.0, "Processing complete")
                
                total_words = sum(page['word_count'] for page in page_data.values())
                self.logger.info(
                    f"Successfully processed document {doc_id}\n"
                    f"Total words: {total_words}\n"
                    f"Pages processed: {len(page_data)}\n"
                    f"Annotations found: {len(annotations)}"
                )
                
                return True

        except Exception as e:
            error_message = f"Error processing document: {str(e)}"
            self.logger.error(f"Error in processing document {doc_id}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            db_manager.update_processing_progress(doc_id, 0.0, error_message)
            return False

    def _transcribe_text(self, image: Image.Image) -> str:
        """First pass: Just get the basic transcription without any special processing."""
        try:
            self.logger.info(f"Transcribing image of size {image.width}x{image.height}")
            self._log_memory_usage("Before transcription")
            
            # Create message with a simple transcription prompt
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
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process vision information with the utility function
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Prepare inputs with the processor
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                input_ids = inputs.input_ids
                
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
                ]
                
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )
                
                text = output_text[0]
            
            # Clean up resources
            del inputs, generated_ids, generated_ids_trimmed
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            text_preview = text[:100] + "..." if text else "[No text recognized]"
            self.logger.info(f"Transcription preview: {text_preview}")
            
            return text

        except Exception as e:
            self.logger.error(f"Error transcribing text: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return ""

    def _detect_boxed_text(self, image: Image.Image, page_num: int) -> list:
        """Second pass: Detect text that has been boxed/highlighted."""
        try:
            self.logger.info(f"Detecting boxed text in image for page {page_num}")
            
            # Force memory cleanup before starting
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                import gc
                gc.collect()
            
            self._log_memory_usage("Start of annotation detection (after cleanup)")
            
            # Explicitly use torch.cuda.amp for mixed precision
            with torch.cuda.amp.autocast(enabled=True):
                # Use a shorter prompt
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
                            {"type": "text", "text": "Find text in boxes."},
                        ],
                    }
                ]
                
                # Track token counts to understand sequence length differences
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                tokenized = self.tokenizer(text, return_tensors="pt")
                input_token_count = tokenized.input_ids.shape[1]
                self.logger.info(f"Input token count: {input_token_count}")
                
                image_inputs, video_inputs = process_vision_info(messages)
                
                # Use smaller batch size
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                # Force to GPU as INT8
                for k, v in inputs.items():
                    if hasattr(v, 'to'):
                        inputs[k] = v.to(self.device)
                
                self._log_memory_usage("Before generation")
                
                # Use explicit memory-saving parameters
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=100,  # Reduced significantly
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,  # Explicitly enable KV caching
                        repetition_penalty=1.0,
                        low_memory=True,  # Some models support this option
                        max_length=input_token_count + 100  # Explicitly limit max length
                    )
                    
                    self._log_memory_usage("After generation")
                    
                    # Trim input tokens from output
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    
                    # Decode output
                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False
                    )
                    
                    response = output_text[0]
            
            # Immediate aggressive cleanup
            del inputs, generated_ids, generated_ids_trimmed, tokenized
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                import gc
                gc.collect()
            
            self._log_memory_usage("After cleanup")
            
            # Parse response to find boxed text - simple line-based approach
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            annotations = []
            for line in lines:
                if line and len(line) > 3:  # Basic filtering
                    annotations.append({
                        'page_number': page_num,
                        'annotation_type': 'box',
                        'text': line.strip(),
                        'confidence': 0.8
                    })
            
            self.logger.info(f"Found {len(annotations)} boxed text items on page {page_num}")
            return annotations
            
        except Exception as e:
            self.logger.error(f"Error detecting boxed text: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _process_image(self, image: Image.Image) -> str:
        """Process a single image with Qwen2.5-VL using the approach from model_test.py."""
        try:
            self.logger.info(f"Processing image of size {image.width}x{image.height}")
            self._log_memory_usage("Before image processing")
            
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
                        {"type": "text", "text": "Transcribe the handwritten text in this image as exactly as possible. At the bottom of your transcription, create a line of ten (10) hyphens. After the hyphens, if you have seen a symbol that looks like a circle with an X through it on a line with text, of roughly the same size as the letters in that line, write CALLOUT: and then that line of text. Only write CALLOUT: and then the line if that specific line had the circle and X character in it. Do this for each instance of a circle with an X through it on the page."},
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

            return text

        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return ""