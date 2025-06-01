import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import tempfile
import time

import torch
from transformers import AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5VLForCausalLM
from PIL import Image
from pdf2image import convert_from_path
from db_manager import DatabaseManager

Image.MAX_IMAGE_PIXELS = 200000000

class QwenVLProcessor:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", use_cache: bool = True):
        os.environ['TRANSFORMERS_CACHE'] = '/mnt/rectangularfile/qwencache'
        
        self.setup_logging()
        self.model_name = model_name
        self.use_cache = use_cache
        
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.logger.info(f"Using GPU: {gpu_name} with {gpu_memory:.2f} GB VRAM")
        else:
            self.device = "cpu"
            self.logger.info("No GPU detected, using CPU")
        
        self.tokenizer = None
        self.processor = None
        self.model = None
    
    def setup_logging(self):
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
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            self.logger.info(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    def _load_model(self):
        if self.model is None:
            try:
                self.logger.info(f"Loading Qwen2.5-VL model: {self.model_name}")
                start_time = time.time()
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True
                )
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                if self.device == "cuda":
                    self.logger.info("Loading model with INT8 quantization")
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                    self.model = Qwen2_5VLForCausalLM.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        trust_remote_code=True,
                        quantization_config=quantization_config
                    )
                else:
                    self.logger.info("Loading model for CPU inference")
                    self.model = Qwen2_5VLForCausalLM.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                
                self.model.eval()
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
        dpi: int = 150
    ) -> bool:
        try:
            self.logger.info(f"=== Starting Qwen VL processing for document {doc_id} ===")
            self.logger.info(f"Processing file: {pdf_path}")
            
            if not self._load_model():
                self.logger.error("Failed to load Qwen2.5-VL model")
                db_manager.update_processing_progress(
                    doc_id, 0.0, "Failed to load Qwen2.5-VL model"
                )
                return False
            
            if not db_manager.mark_ocr_started(doc_id):
                self.logger.error(f"Failed to mark document {doc_id} for processing")
                return False
            
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
                    
                    processed_images = []
                    max_pixels = 3000000 if self.device == "cuda" else 1000000
                    
                    for img in images:
                        if img.width * img.height > max_pixels:
                            scale_factor = (max_pixels / (img.width * img.height)) ** 0.5
                            new_width = int(img.width * scale_factor)
                            new_height = int(img.height * scale_factor)
                            
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
        try:
            self.logger.info(f"Processing image of size {image.width}x{image.height}")
            
            prompt = """<|im_start|>system
You are a helpful assistant that accurately transcribes handwritten text from images.
<|im_end|>
<|im_start|>user
Please transcribe all handwritten text in this image, preserving layout and line breaks.
<|im_end|>
<|im_start|>assistant
I'll transcribe the handwritten text from the image, maintaining its layout:
"""
            
            self.logger.info("Preparing inputs with processor...")
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True
            )
            
            for key, value in inputs.items():
                if hasattr(value, 'shape'):
                    self.logger.info(f"- {key}: shape {value.shape}")
                elif isinstance(value, list):
                    self.logger.info(f"- {key}: list of length {len(value)}")
                else:
                    self.logger.info(f"- {key}: type {type(value)}")
            
            inputs = inputs.to(self.device)
            self.logger.info(f"Moved inputs to {self.device}")
            
            self._log_memory_usage()
            
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
            
            self.logger.info("Generation completed")
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            del inputs, outputs
            if self.device == "cuda":
                torch.cuda.empty_cache()
                self._log_memory_usage()
            
            response_start = generated_text.find("<|im_start|>assistant")
            if response_start != -1:
                text_start = generated_text.find("\n", response_start) + 1
                text_end = generated_text.find("<|im_end|>", text_start)
                if text_end == -1:
                    text_end = None
                text = generated_text[text_start:text_end].strip()
            else:
                text = generated_text.strip()
            
            text_preview = text[:100] + "..." if text else "[No text recognized]"
            self.logger.info(f"Recognized text preview: {text_preview}")
            
            return text, 0.95 if text else 0.0
                
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return "", 0.0