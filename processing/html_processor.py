# processing/html_processor.py
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict
import os
import re
from bs4 import BeautifulSoup
from db.db_manager import DatabaseManager

class HTMLProcessor:
    """Process HTML text notes from e-ink devices."""
    
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
    
    def process_document(self, html_path: Path, doc_id: int, db_manager: DatabaseManager) -> bool:
        """Process an HTML document and extract its text content."""
        try:
            self.logger.info(f"Processing HTML document: {html_path}")
            
            if not html_path.exists():
                self.logger.error(f"HTML file not found: {html_path}")
                db_manager.update_processing_progress(doc_id, 0.0, "HTML file not found")
                return False
                
            if not db_manager.mark_text_extraction_started(doc_id):
                self.logger.error("Failed to mark document as processing")
                return False
            
            db_manager.update_processing_progress(doc_id, 10.0, "Extracting HTML content")
            
            # Read the HTML file with error handling
            try:
                with open(html_path, 'r', encoding='utf-8', errors='replace') as f:
                    html_content = f.read()
            except Exception as e:
                self.logger.error(f"Error reading HTML file: {e}")
                db_manager.update_processing_progress(doc_id, 0.0, f"Error reading HTML: {str(e)}")
                return False
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title if available
            title = soup.title.string if soup.title else os.path.basename(html_path)
            self.logger.info(f"Title: {title}")
            
            # Extract body text, removing script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text and normalize whitespace
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            self.logger.info(f"Extracted {len(text)} characters of text from HTML")
            
            # Calculate word count
            words = re.findall(r'\w+', text)
            word_count = len(words)
            self.logger.info(f"Word count: {word_count}")
            
            # Update progress
            db_manager.update_processing_progress(doc_id, 50.0, "Processing content")
            
            # Prepare page data (HTML notes are treated as single-page documents)
            page_data = {
                1: {
                    'text': text,
                    'word_count': word_count,
                    'char_count': len(text),
                    'processed_at': datetime.now()
                }
            }
            
            # Store the extracted text
            db_manager.update_processing_progress(doc_id, 80.0, "Storing content")
            success = db_manager.store_extracted_text(doc_id, page_data)
            
            if success:
                self.logger.info(f"Successfully processed HTML document {doc_id}")
                return True
            else:
                self.logger.error(f"Failed to store text for document {doc_id}")
                db_manager.update_processing_progress(doc_id, 0.0, "Failed to store content")
                return False
                
        except Exception as e:
            error_message = f"Error processing HTML document: {str(e)}"
            self.logger.error(f"Error in process_document: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            db_manager.update_processing_progress(doc_id, 0.0, error_message)
            return False
            
    def _extract_metadata(self, soup: BeautifulSoup, html_path: Path) -> Dict:
        """
        Extract metadata from the HTML document.
        This is device-specific and can be extended for different devices.
        """
        metadata = {}
        
        # Extract filename metadata (date patterns, etc.)
        filename = os.path.basename(html_path)
        import re
        date_match = re.match(r'^(\d{4})(\d{2})(\d{2})[ _-]?(.*)', filename)
        if date_match:
            year, month, day, remaining = date_match.groups()
            try:
                # Validate the date
                datetime(int(year), int(month), int(day))
                metadata['date'] = f"{year}-{month}-{day}"
                # Use the remaining part for title
                remaining = os.path.splitext(remaining)[0]
                metadata['title'] = remaining.replace('_', ' ').replace('-', ' ').strip()
            except ValueError:
                # If date is invalid, just use the filename
                metadata['title'] = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').strip()
        else:
            # No date pattern, just use filename
            metadata['title'] = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').strip()
        
        # Boox-specific metadata extraction
        # Look for metadata in meta tags or specific div structures
        
        # Example: extract creation date from meta tags
        meta_date = soup.find('meta', {'name': 'date'})
        if meta_date and 'content' in meta_date.attrs:
            metadata['created_at'] = meta_date['content']
        
        # Example: extract author if available
        meta_author = soup.find('meta', {'name': 'author'})
        if meta_author and 'content' in meta_author.attrs:
            metadata['author'] = meta_author['content']
        
        return metadata