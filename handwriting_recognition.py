import os
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import tempfile
from datetime import datetime

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from db_manager import DatabaseManager

class HandwritingRecognizer:
    """Process PDF documents for handwriting recognition."""
    
    def __init__(self, tesseract_config: Optional[Dict[str, str]] = None):
        """
        Initialize the handwriting recognizer.
        
        Args:
            tesseract_config: Optional configuration for Tesseract OCR
        """
        self.setup_logging()
        
        # Default Tesseract configuration for handwriting recognition
        self.tesseract_config = tesseract_config or {
            'lang': 'eng',  # Language (eng = English)
            'config': '--psm 6 --oem 1',  # Page segmentation mode: 6 = Assume a single block of text
                                      # OCR Engine Mode: 1 = Neural nets LSTM only
        }
        
        # Ensure Tesseract is installed and available
        try:
            version = pytesseract.get_tesseract_version()
            self.logger.info(f"Tesseract OCR version {version} detected")

            # Check if we can run a simple test
            self.logger.info("Testing Tesseract availability...")
            test_result = pytesseract.get_languages()
            self.logger.info(f"Available Tesseract languages: {test_result}")
        except Exception as e:
            self.logger.error(f"Error initializing Tesseract OCR: {e}")
            self.logger.error("Make sure Tesseract is installed and in your PATH")
    
    def setup_logging(self):
        """Configure logging for OCR operations."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def process_document(
        self, 
        pdf_path: Path, 
        doc_id: int, 
        db_manager: DatabaseManager,
        dpi: int = 300
    ) -> bool:
        """
        Process a PDF document for handwriting recognition.
        
        Args:
            pdf_path: Path to the PDF file
            doc_id: Database ID of the document
            db_manager: Database manager instance
            dpi: DPI for PDF to image conversion (higher = better quality but slower)
            
        Returns:
            bool: True if processing was successful
        """
        try:
            self.logger.info(f"=== Starting handwriting recognition for document {doc_id} ===")
            self.logger.info(f"Processing file: {pdf_path}")
            
            # Mark document as being processed
            if not db_manager.mark_ocr_started(doc_id):
                self.logger.error(f"Failed to mark document {doc_id} for OCR processing")
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
                        fmt='jpg',
                        grayscale=True
                    )
                    self.logger.info(f"Converted PDF to {len(images)} images")
                except Exception as pdf_error:
                    self.logger.error(f"Error converting PDF to images: {pdf_error}")
                    self.logger.exception("Full conversion traceback:")
                    db_manager.update_processing_progress(
                        doc_id, 0.0, f"Error converting PDF to images: {str(pdf_error)}"
                    )
                    return False

                if not images:
                    self.logger.warning(f"No images extracted from PDF {doc_id}")
                    db_manager.update_processing_progress(
                        doc_id, 0.0, "No images could be extracted from PDF"
                    )
                    return False

                # Process each page
                page_data = {}
                progress_per_page = 80.0 / len(images)

                for i, image in enumerate(images):
                    page_num = i + 1
                    current_progress = 20.0 + (i * progress_per_page)

                    self.logger.info(f"Processing page {page_num}/{len(images)}")
                    db_manager.update_processing_progress(
                        doc_id,
                        current_progress,
                        f"Recognizing handwriting on page {page_num}/{len(images)}"
                    )

                    # Save intermediate image for debugging
                    debug_path = os.path.join(temp_dir, f"debug_page_{page_num}.jpg")
                    image.save(debug_path)
                    self.logger.info(f"Saved debug image to {debug_path}")

                    # Preprocess the image for better recognition
                    processed_image = self._preprocess_image(image)

                    # Save processed image for debugging
                    processed_path = os.path.join(temp_dir, f"processed_page_{page_num}.jpg")
                    processed_image.save(processed_path)
                    self.logger.info(f"Saved processed image to {processed_path}")

                    # Perform handwriting recognition
                    text, confidence, word_count = self._recognize_handwriting(processed_image)

                    # Log sample of recognized text
                    text_sample = text[:100].replace('\n', ' ') if text else "[No text recognized]"
                    self.logger.info(f"Text sample: {text_sample}...")

                    char_count = len(text)

                    # Store the OCR result
                    page_data[page_num] = {
                        'text': text,
                        'confidence': confidence,
                        'word_count': word_count,
                        'char_count': char_count,
                        'processed_at': datetime.now()
                    }

                # Store OCR results in database
                self.logger.info(f"Storing OCR results for {len(page_data)} pages")
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
                    self.logger.error(f"Failed to store OCR results for document {doc_id}")
                    db_manager.update_processing_progress(
                        doc_id, 0.0, "Failed to store OCR results"
                    )

                return success

        except Exception as e:
            error_message = f"Error processing document: {str(e)}"
            self.logger.error(f"Error in handwriting recognition for document {doc_id}: {e}")
            self.logger.exception("Full traceback:")
            db_manager.update_processing_progress(doc_id, 0.0, error_message)
            return False
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better handwriting recognition.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Processed PIL Image
        """
        self.logger.info("Preprocessing image for handwriting recognition")
        from PIL import ImageEnhance, ImageFilter
        
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
            
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Apply mild blur to reduce noise
        image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        
        return image

    def _recognize_handwriting(self, image: Image.Image) -> Tuple[str, float, int]:
        """
        Perform handwriting recognition on an image.

        Args:
            image: Preprocessed PIL Image

        Returns:
            Tuple of (recognized_text, confidence, word_count)
        """
        self.logger.info("Performing handwriting recognition")

        # First check if we have a trained model
        trained_model_path = None
        try:
            from handwriting_trainer import HandwritingTrainer
            trainer = HandwritingTrainer(self.db_manager if hasattr(self, 'db_manager') else None)
            trained_model_path = trainer.get_trained_model_path("default")
        except Exception as e:
            self.logger.warning(f"Error checking for trained model: {e}")

        # Configure for handwriting recognition
        config = '--psm 6 --oem 1'

        # If we have a trained model, use it
        tessdata_dir = None
        if trained_model_path and os.path.exists(trained_model_path):
            self.logger.info(f"Using trained model: {trained_model_path}")
            # Extract parent directory from model path
            tessdata_dir = os.path.dirname(trained_model_path)
            # Add TESSDATA_PREFIX to environment for this process
            os.environ['TESSDATA_PREFIX'] = tessdata_dir
            self.logger.info(f"Set TESSDATA_PREFIX to {tessdata_dir}")

        # Try to recognize handwriting
        try:
            # Recognition with model if available
            text = pytesseract.image_to_string(
                image,
                lang='eng',  # You'd use a custom language code for your trained model
                config=config
            )

            # Get detailed OCR data including confidence
            ocr_data = pytesseract.image_to_data(
                image,
                lang='eng',
                config=config,
                output_type=pytesseract.Output.DICT
            )

            # Calculate confidence score (average of word confidences)
            confidences = [conf for conf in ocr_data['conf'] if conf != -1]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            avg_confidence = avg_confidence / 100.0  # Convert to 0-1 scale

            word_count = len([word for word in ocr_data['text'] if word.strip()])

            self.logger.info(f"Recognition complete: {word_count} words, confidence: {avg_confidence:.2f}")

            # Clean up environment if we set it
            if tessdata_dir:
                if 'TESSDATA_PREFIX' in os.environ:
                    del os.environ['TESSDATA_PREFIX']

            return text, avg_confidence, word_count

        except Exception as e:
            self.logger.error(f"Error in handwriting recognition: {e}")
            return "", 0.0, 0
