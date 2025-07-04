# utils/helpers.py
from typing import Dict
import re
from datetime import datetime
import os

def extract_filename_metadata(filename: str) -> Dict[str, str]:
    """
    Extract useful metadata from filename patterns.
    For example: 20250514_Meeting_Name.pdf -> {'date': '2025-05-14', 'title': 'Meeting Name'}
    """
    metadata = {}
    
    # Extract date if filename starts with a date pattern (YYYYMMDD)
    date_match = re.match(r'^(\d{4})(\d{2})(\d{2})[ _-]?(.*)', filename)
    if date_match:
        year, month, day, remaining = date_match.groups()
        try:
            # Validate the date
            datetime(int(year), int(month), int(day))
            metadata['date'] = f"{year}-{month}-{day}"
            # Use the remaining part for title
            metadata['title'] = remaining.replace('_', ' ').replace('-', ' ')
        except ValueError:
            # If date is invalid, just use the whole filename
            metadata['title'] = filename.replace('.pdf', '')
    else:
        # No date pattern, just use the filename without extension
        metadata['title'] = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ')
    
    return metadata

def calculate_processing_progress(doc: dict) -> float:
    """Calculate the processing progress percentage for a document."""
    status = doc.get('processing_status', '').lower()
    if status == 'completed':
        return 100.0
    elif status == 'failed':
        return 0.0
    elif status == 'pending':
        return 0.0
    elif status == 'processing':
        return doc.get('processing_progress', 50.0)
    else:
        return 0.0