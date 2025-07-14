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
from .cv_annotation_detector import CVAnnotationDetector

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
        
        # Initialize CV annotation detector
        self.cv_detector = CVAnnotationDetector()

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
                    
                    # Create CalDAV todos from yellow highlights
                    self.logger.info("Attempting to create CalDAV todos from highlights")
                    self._create_todos_from_highlights(annotations, db_manager)
                
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
        """Hybrid approach: Use CV to detect colored regions, then VLM to extract text."""
        annotations_found: List[Dict] = []
        
        try:
            self.logger.info(f"Detecting colored annotations on page {page_num} using hybrid CV+VLM approach")
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                import gc
                gc.collect()
            
            # Step 1: Use computer vision to detect colored regions
            cv_detections = self.cv_detector.detect_all_annotations(image)
            
            # Step 2: Extract text from detected regions using VLM
            for annotation_type, regions in cv_detections.items():
                for region in regions:
                    # Extract the region image
                    region_image = self.cv_detector.extract_region_image(image, region['bbox'])
                    
                    # Use VLM to extract text from this specific region
                    extracted_text = self._extract_text_from_region(region_image, region['type'])
                    
                    if extracted_text and extracted_text.strip():
                        annotations_found.append({
                            'page_number': page_num,
                            'annotation_type': region['type'],
                            'text': extracted_text.strip(),
                            'confidence': 0.8  # Fixed value for database compatibility
                        })
                    
                    # Clean up between regions
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
            
            self.logger.info(f"Found {len(annotations_found)} total annotations using hybrid approach")
            return annotations_found
            
        except Exception as e:
            self.logger.error(f"Error detecting colored annotations: {e}")
            return []
    
    def _extract_text_from_region(self, region_image: Image.Image, region_type: str) -> str:
        """Extract text from a specific region using VLM with simplified prompt."""
        try:
            self.logger.info(f"Extracting text from {region_type} region of size {region_image.width}x{region_image.height}")
            
            # Simple prompt focused on text extraction
            prompt = "Extract all handwritten text from this image region. Transcribe exactly what you see, ignoring any colored backgrounds or boxes."
            
            messages = [{"role": "user", "content": [
                {"type": "image", "image": region_image, "resized_height": region_image.height, "resized_width": region_image.width}, 
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
            
            with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                with torch.no_grad():
                    current_input_ids = inputs_dict_on_device['input_ids']
                    generated_ids = self.model.generate(
                        **inputs_dict_on_device,
                        max_new_tokens=200,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(current_input_ids, generated_ids)]
                    output_text_list = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    extracted_text = output_text_list[0] if output_text_list else ""
            
            del inputs_dict, inputs_dict_on_device, generated_ids, generated_ids_trimmed, current_input_ids
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            self.logger.info(f"Extracted text: {extracted_text[:50]}..." if len(extracted_text) > 50 else f"Extracted text: {extracted_text}")
            return extracted_text
            
        except Exception as e:
            self.logger.error(f"Error extracting text from region: {e}")
            return ""
    
    def _create_todos_from_highlights(self, annotations: List[Dict], db_manager) -> None:
        """Create CalDAV todos from yellow highlight annotations."""
        try:
            self.logger.info(f"_create_todos_from_highlights called with {len(annotations)} annotations")
            
            # Check if CalDAV is enabled
            settings = db_manager.get_caldav_settings()
            self.logger.info(f"CalDAV settings retrieved: enabled={settings['enabled']}, url={bool(settings['url'])}, username={bool(settings['username'])}, password={bool(settings['password'])}")
            
            if not settings['enabled']:
                self.logger.info("CalDAV integration disabled, skipping todo creation")
                return
            
            if not all([settings['url'], settings['username'], settings['password']]):
                self.logger.warning(f"CalDAV settings incomplete, skipping todo creation. url={bool(settings['url'])}, username={bool(settings['username'])}, password={bool(settings['password'])}")
                return
            
            # Filter for yellow highlights only  
            yellow_highlights = [ann for ann in annotations if ann.get('annotation_type') == 'yellow_highlight']
            self.logger.info(f"Filtered annotations: {len(yellow_highlights)} yellow highlights out of {len(annotations)} total")
            
            if not yellow_highlights:
                self.logger.info("No yellow highlights found, skipping todo creation")
                return
            
            self.logger.info(f"Creating {len(yellow_highlights)} todos from highlights")
            for i, highlight in enumerate(yellow_highlights):
                self.logger.info(f"Highlight {i+1}: type={highlight.get('annotation_type')}, text='{highlight.get('text', '')[:50]}...'")
                
            
            # Import CalDAV client
            from processing.caldav_client import CalDAVTodoClient
            client = CalDAVTodoClient()
            
            # Connect to CalDAV server
            if not client.connect(settings['url'], settings['username'], 
                                  settings['password'], settings['calendar']):
                self.logger.error("Failed to connect to CalDAV server for todo creation")
                return
            
            created_count = 0
            for highlight in yellow_highlights:
                try:
                    # Extract text from highlight (should already be in annotation)
                    text = highlight.get('text', '').strip()
                    
                    if not text:
                        self.logger.warning(f"No text found for highlight on page {highlight.get('page_number', 'unknown')}")
                        continue
                    
                    # Clean up text for CalDAV compatibility
                    # Replace newlines with spaces and normalize whitespace
                    cleaned_text = ' '.join(text.split())
                    
                    # Create a meaningful summary (first line or limited chars)
                    summary = cleaned_text[:100] if len(cleaned_text) <= 100 else cleaned_text[:100] + "..."
                    
                    # Use full cleaned text as description if longer than summary
                    description = cleaned_text if len(cleaned_text) > len(summary) else ""
                    
                    # Set categories to identify source
                    categories = ['RectangularFile', 'Handwritten']
                    page_num = highlight.get('page_number')
                    if page_num:
                        categories.append(f'Page-{page_num}')
                    
                    # Create todo with medium priority
                    todo_uid = client.create_todo(
                        summary=summary,
                        description=description,
                        priority=5,  # Medium priority
                        categories=categories
                    )
                    
                    if todo_uid:
                        created_count += 1
                        self.logger.debug(f"Created todo: {summary[:30]}...")
                    else:
                        self.logger.warning(f"Failed to create todo for highlight: {summary[:30]}...")
                        
                except Exception as e:
                    self.logger.error(f"Error creating todo from highlight: {e}")
                    continue
            
            if created_count > 0:
                self.logger.info(f"Successfully created {created_count} todos from highlights")
            else:
                self.logger.warning("No todos were created from highlights")
                
        except Exception as e:
            self.logger.error(f"Error in todo creation from highlights: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

