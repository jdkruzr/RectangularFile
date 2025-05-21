from typing import Dict
from pathlib import Path
import logging
from datetime import datetime
import os

from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer, LTChar, LTPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from db_manager import DatabaseManager  # Add the import for type hints

class PDFProcessor:
    def __init__(self):
        self.setup_logging()
    
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
    
    def check_text_presence(self, pdf_path: Path) -> bool:
        try:
            self.logger.info(f"Checking for text content in {pdf_path}")
            
            with open(pdf_path, 'rb') as file:
                parser = PDFParser(file)
                document = PDFDocument(parser)
                
                if not document.is_extractable:
                    self.logger.info(f"PDF is not extractable: {pdf_path}")
                    return False
                
                some_text = extract_text(pdf_path, maxpages=1)
                has_text = bool(some_text.strip())
                
                self.logger.info(
                    f"Text content {'found' if has_text else 'not found'} in {pdf_path}"
                )
                return has_text
                
        except Exception as e:
            self.logger.error(f"Error checking text presence in {pdf_path}: {e}")
            return False
    
    def get_page_count(self, pdf_path: Path) -> int:
        try:
            with open(pdf_path, 'rb') as file:
                parser = PDFParser(file)
                document = PDFDocument(parser)
                return len(document.catalog['Pages'].resolve()['Kids'])
        except Exception as e:
            self.logger.error(f"Error getting page count: {e}")
            return 0

    def process_document(
        self,
        pdf_path: Path,
        doc_id: int,
        db_manager: DatabaseManager
    ) -> bool:
        try:
            self.logger.info(f"Starting text extraction for document {doc_id}")

            if not db_manager.mark_text_extraction_started(doc_id):
                self.logger.error("Failed to mark document as processing")
                return False

            self.logger.info("Checking for extractable text...")
            db_manager.update_processing_progress(
                doc_id, 10.0, "Checking for extractable text"
            )

            if not self.check_text_presence(pdf_path):
                self.logger.info(f"No extractable text found in document {doc_id}")
                db_manager.update_processing_progress(
                    doc_id, 0.0, "No extractable text found"
                )
                return False

            self.logger.info("Text found, beginning extraction...")
            db_manager.update_processing_progress(
                doc_id, 20.0, "Beginning text extraction"
            )

            self.logger.info("Getting page count...")
            total_pages = self.get_page_count(pdf_path)
            self.logger.info(f"Found {total_pages} pages")
            
            if total_pages == 0:
                self.logger.warning(f"No pages found in document {doc_id}")
                db_manager.update_processing_progress(
                    doc_id, 0.0, "Document appears to be empty"
                )
            return False

            page_data = {}
            progress_per_page = 60.0 / total_pages if total_pages > 0 else 60.0

            self.logger.info("Starting page-by-page extraction...")
            for page_num, page_layout in enumerate(extract_pages(pdf_path), 1):
                self.logger.info(f"Processing page {page_num}/{total_pages}")
                
                if not isinstance(page_layout, LTPage):
                    self.logger.warning(f"Skipping non-page element on page {page_num}")
                    continue

                current_progress = 20.0 + (page_num * progress_per_page)
                db_manager.update_processing_progress(
                    doc_id,
                    current_progress,
                    f"Extracting text from page {page_num}/{total_pages}"
                )

                page_text = []
                char_count = 0

                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        text = element.get_text()
                        page_text.append(text)
                        char_count += sum(
                            1 for char in element
                            if isinstance(char, LTChar)
                        )

                text = ''.join(page_text).strip()
                word_count = len(text.split())
                confidence = 1.0 if char_count > 0 else 0.0

                self.logger.info(f"Page {page_num}: found {word_count} words")
                
                page_data[page_layout.pageid] = {
                    'text': text,
                    'confidence': confidence,
                    'word_count': word_count,
                    'char_count': char_count,
                    'processed_at': datetime.now()
                }

            if not page_data:
                self.logger.warning(f"No text extracted from document {doc_id}")
                db_manager.update_processing_progress(
                    doc_id, 0.0, "No text could be extracted"
                )
                return False

            self.logger.info("Storing extracted text...")
            db_manager.update_processing_progress(
                doc_id, 80.0, "Storing extracted text"
            )

            success = db_manager.store_extracted_text(doc_id, page_data)

            if success:
                self.logger.info(f"Successfully processed document {doc_id}")
            else:
                self.logger.error(f"Failed to store text for document {doc_id}")
                db_manager.update_processing_progress(
                    doc_id, 0.0, "Failed to store extracted text"
                )

            return success

        except Exception as e:
            error_message = f"Error processing document: {str(e)}"
            self.logger.error(f"Error processing document {doc_id}: {e}")
            db_manager.update_processing_progress(doc_id, 0.0, error_message)
            return False
