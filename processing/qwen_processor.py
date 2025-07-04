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
    # BitsAndBytesConfig, # Not using for FP16
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
    """Process PDF documents using Qwen2.5-VL-7B-Instruct for text recognition."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", use_cache: bool = True):
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
            # Calculate free memory based on total - reserved (more accurate for available)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free = total_memory - reserved


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
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_path_str = str(debug_dir / f"original_{timestamp}_doc_{doc_id}_page_{page_num}.jpg")
        try:
            image.save(original_path_str)
            self.logger.info(f"Saved original image to {original_path_str}")
        except Exception as e:
            self.logger.error(f"Error saving original image {original_path_str}: {e}")
            original_path_str = "" 


        original_pixels = image.width * image.height
        if original_pixels <= self.target_image_pixels:
            return image, original_path_str

        ratio = (self.target_image_pixels / original_pixels) ** 0.5
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)

        self.logger.info(
            f"Resizing image from {image.width}x{image.height} "
            f"({original_pixels:,} pixels) to {new_width}x{new_height} "
            f"({new_width * new_height:,} pixels) to conserve memory"
        )

        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        resized_path_str = str(debug_dir / f"resized_{timestamp}_doc_{doc_id}_page_{page_num}.jpg")
        try:
            resized_image.save(resized_path_str)
            self.logger.info(f"Saved resized image to {resized_path_str}")
        except Exception as e:
            self.logger.error(f"Error saving resized image {resized_path_str}: {e}")

        return resized_image, original_path_str

    def _load_model(self):
        """Load the model with INT8 quantization optimized for 16GB VRAM."""
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

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                if self.device == "cuda":
                    # Import BitsAndBytes for quantization
                    from transformers import BitsAndBytesConfig
                    
                    self.logger.info("Loading 7B model with INT8 quantization for 16GB VRAM")
                    
                    # Configure quantization for memory efficiency
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False
                    )
                    
                    # Set conservative memory limits for 16GB card
                    memory_map = {0: "12GB"}  # Reserve ~4GB for other operations
                    
                    model_kwargs = {
                        "device_map": "auto",
                        "trust_remote_code": True,
                        "quantization_config": quantization_config,
                        "max_memory": memory_map,
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

                self._log_memory_usage("After model loading")
                elapsed = time.time() - start_time
                self.logger.info(f"7B model loaded successfully in {elapsed:.2f} seconds")
                return True

            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return False
        return True

    def process_document(self, pdf_path: Path, doc_id: int, db_manager: DatabaseManager, dpi: int = 150, detect_annotations: bool = True) -> bool:
        # Initialize annotations list at the beginning of the method scope
        annotations: List[Dict] = [] 
                
        try: # Main try block for the entire method
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
                try:
                    images_pil_list = convert_from_path(
                        pdf_path,
                        dpi=dpi,
                        output_folder=temp_dir,
                        fmt='jpg'
                    )
                    self.logger.info(f"Converted PDF to {len(images_pil_list)} images")
                except Exception as pdf_error:
                    self.logger.error(f"Error converting PDF to images: {pdf_error}")
                    db_manager.update_processing_progress(doc_id, 0.0, f"Error converting PDF: {str(pdf_error)}")
                    return False

                if not images_pil_list:
                    self.logger.warning(f"No images extracted from PDF {doc_id}")
                    db_manager.update_processing_progress(doc_id, 0.0, "No images from PDF")
                    return False

                page_data = {}
                progress_per_page = 60.0 / len(images_pil_list) if images_pil_list else 60.0
                resized_images_for_processing: List[Image.Image] = []

                self.logger.info("Starting first pass: Text transcription")
                for i, image_pil_original in enumerate(images_pil_list):
                    page_num = i + 1
                    current_progress = 20.0 + (i * progress_per_page)
                    self.logger.info(f"Transcribing page {page_num}/{len(images_pil_list)}")
                    db_manager.update_processing_progress(doc_id, current_progress, f"Transcribing page {page_num}")

                    image_for_transcription, image_path_str = self._resize_image_if_needed(image_pil_original, doc_id, page_num)
                    resized_images_for_processing.append(image_for_transcription)
                    
                    text_content = self._transcribe_text(image_for_transcription)
                    
                    # Corrected dictionary assignment
                    page_data[page_num] = { 
                        'text': text_content,
                        'word_count': len(text_content.split()) if text_content else 0,
                        'char_count': len(text_content) if text_content else 0,
                        'processed_at': datetime.now(),
                        'image_path': image_path_str
                    }
                    if self.device == "cuda": torch.cuda.empty_cache()

                self.logger.info(f"Storing transcription results for {len(page_data)} pages")
                db_manager.update_processing_progress(doc_id, 80.0, "Storing transcriptions")
                if not db_manager.store_ocr_results(doc_id, page_data):
                    self.logger.error(f"Failed to store transcriptions for doc {doc_id}")
                    db_manager.update_processing_progress(doc_id, 0.0, "Failed to store transcriptions")
                    return False

                if self.device == "cuda":
                    torch.cuda.empty_cache(); torch.cuda.synchronize()
                    self._log_memory_usage("After first pass cleanup")
                    import gc; gc.collect()
                    torch.cuda.empty_cache(); torch.cuda.synchronize()
                    self._log_memory_usage("After garbage collection")

                if not detect_annotations:
                    self.logger.info("Skipping annotation detection.")
                    db_manager.update_processing_progress(doc_id, 100.0, "Processing complete (no annotations)")
                    total_words = sum(p.get('word_count', 0) for p in page_data.values())
                    self.logger.info(
                        f"Successfully processed document {doc_id}\n"
                        f"Total words: {total_words}\n"
                        f"Pages processed: {len(page_data)}\n"
                        f"Annotations detection skipped"
                    )
                    return True

                self.logger.info("Starting second pass: Detecting colored annotations")
                db_manager.update_processing_progress(doc_id, 85.0, "Detecting annotations")
                self._ensure_annotations_table(db_manager)

                for i, image_for_annotations in enumerate(resized_images_for_processing):
                    page_num = i + 1
                    self.logger.info(f"Detecting colored annotations on page {page_num}/{len(resized_images_for_processing)}")
                    page_annotations = self._detect_colored_annotations(image_for_annotations, page_num)
                    annotations.extend(page_annotations)
                    if self.device == "cuda": torch.cuda.empty_cache()
                
                if annotations:
                    self.logger.info(f"Storing {len(annotations)} annotations")
                    db_manager.store_document_annotations(doc_id, annotations)
                
                db_manager.update_processing_progress(doc_id, 100.0, "Processing complete")
                total_words = sum(p.get('word_count', 0) for p in page_data.values())
                self.logger.info(
                    f"Successfully processed document {doc_id}\n"
                    f"Total words: {total_words}\n"
                    f"Pages processed: {len(page_data)}\n"
                    f"Annotations found: {len(annotations)}"
                )
                return True

        except Exception as e: # Matching except for the main try block
            error_message = f"Error processing document {doc_id}: {str(e)}"
            self.logger.error(error_message)
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            db_manager.update_processing_progress(doc_id, 0.0, error_message)
            return False

    def _transcribe_text(self, image: Image.Image) -> str:
        try:
            self.logger.info(f"Transcribing image of size {image.width}x{image.height} (INT8)")
            self._log_memory_usage("Before transcription (INT8)")
            
            messages = [{"role": "user", "content": [{"type": "image", "image": image, "resized_height": image.height, "resized_width": image.width}, {"type": "text", "text": "Transcribe the handwritten text in this image as exactly as possible."}]}]
            
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs_list, video_inputs_list = process_vision_info(messages)
            
            # Corrected: arguments to processor and closing parenthesis
            inputs_dict = self.processor(
                text=[text_prompt], 
                images=image_inputs_list, 
                videos=video_inputs_list, 
                padding=True, 
                return_tensors="pt"
            )
            inputs_dict_on_device = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs_dict.items()}
            
            with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                with torch.no_grad():
                    current_input_ids = inputs_dict_on_device['input_ids']
                    
                    generated_ids = self.model.generate(
                        **inputs_dict_on_device,
                        max_new_tokens=500,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(current_input_ids, generated_ids)]
                    
                    output_text_list = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    transcribed_text = output_text_list[0] if output_text_list else ""
            
            del inputs_dict, inputs_dict_on_device, generated_ids, generated_ids_trimmed, current_input_ids
            if self.device == "cuda": torch.cuda.empty_cache()
                
            self.logger.info(f"Transcription preview (INT8): {transcribed_text[:100] + '...' if transcribed_text else '[No text recognized]'}")
            return transcribed_text

        except Exception as e:
            self.logger.error(f"Error transcribing text (INT8): {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return ""

    def _detect_colored_annotations(self, image: Image.Image, page_num: int) -> list:
        """Detect both green boxed text and yellow highlighted text in separate passes."""
        annotations_found: List[Dict] = []
        
        try:
            self.logger.info(f"Detecting colored annotations on page {page_num}")
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                import gc
                gc.collect()
            
            # Pass 1: Green boxes
            green_annotations = self._detect_green_boxes(image, page_num)
            annotations_found.extend(green_annotations)
            
            # Clean up between passes
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            # Pass 2: Yellow highlights  
            yellow_annotations = self._detect_yellow_highlights(image, page_num)
            annotations_found.extend(yellow_annotations)
            
            self.logger.info(f"Found {len(annotations_found)} total annotations")
            return annotations_found
            
        except Exception as e:
            self.logger.error(f"Error detecting colored annotations: {e}")
            return []

    def _detect_green_boxes(self, image: Image.Image, page_num: int) -> list:
        """Detect only green boxed text."""
        annotations = []
        
        try:
            self.logger.info(f"Detecting green boxes on page {page_num}")
            self._log_memory_usage("Start of green box detection")
            
            max_new_tokens = 100
            
            with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                prompt = """Look for text that has a GREEN BOX drawn around it.

    Requirements:
    - The box must be GREEN colored (not black, blue, or any other color)
    - The box must fully surround the text

    DO NOT include:
    - Text that is simply separated by whitespace
    - Text in black boxes or borders
    - Underlined text
    - Text that stands alone without a green box
    - Section headers or titles without green boxes

    Respond with ONLY the text inside green boxes, one per line.
    If no green boxes found, respond with: NONE"""

                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image, "resized_height": image.height, "resized_width": image.width}, 
                    {"type": "text", "text": prompt}
                ]}]
                
                text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs_list, video_inputs_list = process_vision_info(messages)
                
                inputs_dict = self.processor(
                    text=[text_prompt], 
                    images=image_inputs_list, 
                    videos=video_inputs_list, 
                    padding=True, 
                    return_tensors="pt"
                )
                inputs_dict_on_device = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs_dict.items()}
                
                with torch.no_grad():
                    current_input_ids = inputs_dict_on_device['input_ids']
                    generated_ids = self.model.generate(
                        **inputs_dict_on_device,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        repetition_penalty=1.0
                    )
                    
                    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(current_input_ids, generated_ids)]
                    output_text_list = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    response = output_text_list[0] if output_text_list else ""
            
            del inputs_dict, inputs_dict_on_device, generated_ids, generated_ids_trimmed, current_input_ids
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Parse response
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            for line in lines:
                if line.upper() == "NONE" or not line:
                    continue
                    
                # Skip any explanatory text
                if any(phrase in line.lower() for phrase in ["here is", "i found", "the text", "boxed text"]):
                    continue
                    
                annotations.append({
                    'page_number': page_num,
                    'annotation_type': 'green_box',
                    'text': line,
                    'confidence': 0.8
                })
            
            self.logger.info(f"Found {len(annotations)} green box annotations")
            return annotations
            
        except Exception as e:
            self.logger.error(f"Error detecting green boxes: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def _detect_yellow_highlights(self, image: Image.Image, page_num: int) -> list:
        """Detect only yellow highlighted text."""
        annotations = []
        
        try:
            self.logger.info(f"Detecting yellow highlights on page {page_num}")
            self._log_memory_usage("Start of yellow highlight detection")
            
            max_new_tokens = 100
            
            with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                prompt = """Look for text inside a region of YELLOW highlighting.

    Requirements:
    - The highlight must be YELLOW colored
    - The text must be readable through the yellow highlight

    DO NOT include:
    - Text without any highlighting
    - Text that is simply separated by whitespace
    - Regular text that happens to be NEXT to yellow highlighting
    - Regular text that happens to be isolated

    Respond with ONLY the text that has yellow highlighting, one per line.
    If no yellow highlights found, respond with: NONE"""

                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image, "resized_height": image.height, "resized_width": image.width}, 
                    {"type": "text", "text": prompt}
                ]}]
                
                text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs_list, video_inputs_list = process_vision_info(messages)
                
                inputs_dict = self.processor(
                    text=[text_prompt], 
                    images=image_inputs_list, 
                    videos=video_inputs_list, 
                    padding=True, 
                    return_tensors="pt"
                )
                inputs_dict_on_device = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs_dict.items()}
                
                with torch.no_grad():
                    current_input_ids = inputs_dict_on_device['input_ids']
                    generated_ids = self.model.generate(
                        **inputs_dict_on_device,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        repetition_penalty=1.0
                    )
                    
                    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(current_input_ids, generated_ids)]
                    output_text_list = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    response = output_text_list[0] if output_text_list else ""
            
            del inputs_dict, inputs_dict_on_device, generated_ids, generated_ids_trimmed, current_input_ids
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Parse response
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            for line in lines:
                if line.upper() == "NONE" or not line:
                    continue
                    
                # Skip any explanatory text
                if any(phrase in line.lower() for phrase in ["here is", "i found", "the text", "highlighted text"]):
                    continue
                    
                annotations.append({
                    'page_number': page_num,
                    'annotation_type': 'yellow_highlight',
                    'text': line,
                    'confidence': 0.8
                })
            
            self.logger.info(f"Found {len(annotations)} yellow highlight annotations")
            return annotations
            
        except Exception as e:
            self.logger.error(f"Error detecting yellow highlights: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return []