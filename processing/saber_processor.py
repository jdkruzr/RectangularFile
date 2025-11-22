"""
Saber Note Processor

Parses decrypted Saber note files (.sbn2 BSON format) and extracts
pages, strokes, and metadata for OCR processing.
"""

import struct
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import bson

logger = logging.getLogger(__name__)


class SaberStroke:
    """Represents a single stroke in a Saber note."""

    def __init__(self, stroke_data: Dict[str, Any]):
        """
        Initialize a stroke from BSON stroke data.

        Args:
            stroke_data: Dictionary containing stroke properties from BSON
        """
        self.points: List[Tuple[float, float, float]] = []  # (x, y, pressure)
        self.tool_type: str = stroke_data.get('ty', 'pen')
        self.color: int = int(stroke_data.get('c', 0xFF000000))  # ARGB format
        self.size: float = float(stroke_data.get('s', 1.0))
        self.pen_enabled: bool = stroke_data.get('pe', True)
        self.smoothing: float = float(stroke_data.get('sm', 0.0))
        self.shape: Optional[str] = stroke_data.get('shape')

        # Decode points from binary format
        point_data = stroke_data.get('p', [])
        for point_bytes in point_data:
            if isinstance(point_bytes, bytes) and len(point_bytes) == 12:
                # Each point is 3 little-endian floats: x, y, pressure
                x, y, pressure = struct.unpack('<fff', point_bytes)
                self.points.append((x, y, pressure))

        logger.debug(f"Stroke: {self.tool_type}, {len(self.points)} points, "
                     f"color=0x{self.color:08x}, size={self.size}")

    def get_rgba_color(self) -> Tuple[int, int, int, int]:
        """
        Convert ARGB color to RGBA tuple.

        Returns:
            Tuple of (r, g, b, a) as 0-255 integers
        """
        a = (self.color >> 24) & 0xFF
        r = (self.color >> 16) & 0xFF
        g = (self.color >> 8) & 0xFF
        b = self.color & 0xFF
        return (r, g, b, a)


class SaberPage:
    """Represents a single page in a Saber note."""

    def __init__(self, page_data: Dict[str, Any], page_number: int):
        """
        Initialize a page from BSON page data.

        Args:
            page_data: Dictionary containing page properties from BSON
            page_number: Page index (0-based)
        """
        self.page_number = page_number
        self.width: float = float(page_data.get('w', 1000.0))
        self.height: float = float(page_data.get('h', 1400.0))
        self.strokes: List[SaberStroke] = []

        # Parse strokes
        stroke_data = page_data.get('s', [])
        for stroke_dict in stroke_data:
            try:
                stroke = SaberStroke(stroke_dict)
                self.strokes.append(stroke)
            except Exception as e:
                logger.warning(f"Failed to parse stroke on page {page_number}: {e}")

        logger.info(f"Page {page_number}: {self.width}x{self.height}, "
                    f"{len(self.strokes)} strokes")


class SaberNote:
    """Represents a complete Saber note with all pages and metadata."""

    def __init__(self, bson_data: bytes):
        """
        Initialize a Saber note from decrypted BSON data.

        Args:
            bson_data: Decrypted BSON data from .sbe file

        Raises:
            ValueError: If BSON parsing fails
        """
        # Get document length from first 4 bytes
        if len(bson_data) < 4:
            raise ValueError("BSON data too short")

        doc_length = struct.unpack('<i', bson_data[0:4])[0]

        if doc_length > len(bson_data):
            raise ValueError(f"Invalid BSON document length: {doc_length} > {len(bson_data)}")

        # Decode BSON (only the documented length, ignore trailing bytes)
        try:
            note_data = bson.decode(bson_data[:doc_length])
        except Exception as e:
            raise ValueError(f"Failed to decode BSON: {e}")

        # Parse metadata
        self.version: int = note_data.get('v', 0)
        self.next_image_id: int = note_data.get('ni', 0)
        self.background_color: Optional[int] = note_data.get('b')
        self.background_pattern: str = note_data.get('p', 'plain')
        self.line_height: int = note_data.get('l', 0)
        self.line_thickness: int = note_data.get('lt', 0)
        self.current_page_index: int = note_data.get('c', 0)

        # Parse pages
        self.pages: List[SaberPage] = []
        pages_data = note_data.get('z', [])

        for i, page_dict in enumerate(pages_data):
            try:
                page = SaberPage(page_dict, i)
                self.pages.append(page)
            except Exception as e:
                logger.warning(f"Failed to parse page {i}: {e}")

        logger.info(f"Parsed Saber note: version={self.version}, "
                    f"pages={len(self.pages)}, pattern={self.background_pattern}")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get note metadata as a dictionary.

        Returns:
            Dictionary with note metadata
        """
        return {
            'version': self.version,
            'num_pages': len(self.pages),
            'background_pattern': self.background_pattern,
            'line_height': self.line_height,
            'line_thickness': self.line_thickness,
            'total_strokes': sum(len(page.strokes) for page in self.pages),
        }


class SaberProcessor:
    """Processes Saber notes for OCR pipeline."""

    @staticmethod
    def parse_note(decrypted_data: bytes) -> SaberNote:
        """
        Parse a decrypted Saber note.

        Args:
            decrypted_data: Decrypted BSON data from .sbe file

        Returns:
            Parsed SaberNote object

        Raises:
            ValueError: If parsing fails
        """
        return SaberNote(decrypted_data)

    @staticmethod
    def parse_note_from_file(decrypted_file: Path) -> SaberNote:
        """
        Parse a Saber note from a decrypted file.

        Args:
            decrypted_file: Path to decrypted .sbn2 file

        Returns:
            Parsed SaberNote object

        Raises:
            FileNotFoundError: If file not found
            ValueError: If parsing fails
        """
        if not decrypted_file.exists():
            raise FileNotFoundError(f"Decrypted file not found: {decrypted_file}")

        with open(decrypted_file, 'rb') as f:
            data = f.read()

        return SaberProcessor.parse_note(data)


def test_processor(decrypted_file: str):
    """
    Test utility for processing a decrypted Saber file.

    Args:
        decrypted_file: Path to decrypted file
    """
    processor = SaberProcessor()
    note = processor.parse_note_from_file(Path(decrypted_file))

    print(f"✓ Parsed note: {len(note.pages)} pages")
    print(f"  Metadata: {note.get_metadata()}")

    for page in note.pages:
        print(f"  Page {page.page_number}: {len(page.strokes)} strokes")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python saber_processor.py <decrypted_file>")
        sys.exit(1)

    test_processor(sys.argv[1])
