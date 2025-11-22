"""
Saber Stroke Renderer

Renders Saber note strokes to raster images for OCR processing.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, ImageDraw
import math

from processing.saber_processor import SaberNote, SaberPage, SaberStroke

logger = logging.getLogger(__name__)


class SaberRenderer:
    """Renders Saber notes to raster images."""

    # Default rendering settings
    DEFAULT_DPI = 150  # Resolution for rendering
    BACKGROUND_COLOR = (255, 255, 255)  # White
    LINE_COLOR = (200, 200, 255, 50)  # Light blue, semi-transparent

    def __init__(self, dpi: int = DEFAULT_DPI):
        """
        Initialize the renderer.

        Args:
            dpi: Resolution for rendering (dots per inch)
        """
        self.dpi = dpi

    def render_page(
        self,
        page: SaberPage,
        output_path: Path,
        background_pattern: Optional[str] = None,
        line_height: int = 100,
        line_thickness: int = 3,
    ) -> Path:
        """
        Render a single page to an image file.

        Args:
            page: SaberPage to render
            output_path: Path to save the rendered image
            background_pattern: Background pattern ('lined', 'grid', 'plain', etc.)
            line_height: Height between lines for lined paper
            line_thickness: Thickness of background lines

        Returns:
            Path to the rendered image
        """
        # Create canvas
        width = int(page.width)
        height = int(page.height)

        # Create RGBA image for transparency support
        image = Image.new('RGBA', (width, height), self.BACKGROUND_COLOR + (255,))
        draw = ImageDraw.Draw(image)

        # Draw background pattern
        if background_pattern and background_pattern != 'plain':
            self._draw_background(draw, width, height, background_pattern, line_height, line_thickness)

        # Draw all strokes
        for stroke in page.strokes:
            self._draw_stroke(draw, stroke)

        # Convert to RGB and save
        rgb_image = Image.new('RGB', (width, height), self.BACKGROUND_COLOR)
        rgb_image.paste(image, (0, 0), image)

        rgb_image.save(output_path, 'JPEG', quality=95)
        logger.info(f"Rendered page {page.page_number} to {output_path}")

        return output_path

    def _draw_background(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        pattern: str,
        line_height: int,
        line_thickness: int,
    ):
        """Draw background pattern (lined paper, grid, etc.)."""
        if pattern == 'lined':
            # Draw horizontal lines
            y = line_height
            while y < height:
                draw.line(
                    [(0, y), (width, y)],
                    fill=self.LINE_COLOR,
                    width=max(1, line_thickness // 2)
                )
                y += line_height

        elif pattern == 'grid':
            # Draw horizontal lines
            y = line_height
            while y < height:
                draw.line(
                    [(0, y), (width, y)],
                    fill=self.LINE_COLOR,
                    width=max(1, line_thickness // 2)
                )
                y += line_height

            # Draw vertical lines
            x = line_height
            while x < width:
                draw.line(
                    [(x, 0), (x, height)],
                    fill=self.LINE_COLOR,
                    width=max(1, line_thickness // 2)
                )
                x += line_height

    def _draw_stroke(self, draw: ImageDraw.ImageDraw, stroke: SaberStroke):
        """
        Draw a single stroke on the canvas.

        Args:
            draw: PIL ImageDraw object
            stroke: SaberStroke to render
        """
        if len(stroke.points) < 2:
            return  # Need at least 2 points to draw

        # Get stroke color
        r, g, b, a = stroke.get_rgba_color()
        color = (r, g, b, a)

        points = stroke.points

        # Draw stroke as connected line segments
        # Use pressure to vary line width
        for i in range(len(points) - 1):
            x1, y1, pressure1 = points[i]
            x2, y2, pressure2 = points[i + 1]

            # Calculate line width based on stroke size and pressure
            # Average pressure between the two points
            avg_pressure = (pressure1 + pressure2) / 2.0
            line_width = max(1, int(stroke.size * avg_pressure))

            # Draw line segment
            draw.line(
                [(x1, y1), (x2, y2)],
                fill=color,
                width=line_width
            )

        # Optionally draw circles at each point for smoother appearance
        # This helps with very thick strokes or high-pressure areas
        if stroke.size > 3:
            for x, y, pressure in points:
                radius = max(1, int(stroke.size * pressure / 2))
                bbox = [
                    x - radius, y - radius,
                    x + radius, y + radius
                ]
                draw.ellipse(bbox, fill=color)

    def render_note(
        self,
        note: SaberNote,
        output_dir: Path,
        prefix: str = "page"
    ) -> list[Path]:
        """
        Render all pages of a note to separate image files.

        Args:
            note: SaberNote to render
            output_dir: Directory to save rendered images
            prefix: Prefix for output filenames

        Returns:
            List of paths to rendered images
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        rendered_pages = []

        for page in note.pages:
            output_path = output_dir / f"{prefix}_{page.page_number:03d}.jpg"

            self.render_page(
                page,
                output_path,
                background_pattern=note.background_pattern,
                line_height=note.line_height,
                line_thickness=note.line_thickness,
            )

            rendered_pages.append(output_path)

        return rendered_pages


def test_renderer(decrypted_file: str, output_dir: str):
    """
    Test utility for rendering a decrypted Saber file.

    Args:
        decrypted_file: Path to decrypted file
        output_dir: Directory to save rendered images
    """
    from processing.saber_processor import SaberProcessor

    # Parse the note
    note = SaberProcessor.parse_note_from_file(Path(decrypted_file))
    print(f"✓ Parsed note: {len(note.pages)} pages")

    # Render all pages
    renderer = SaberRenderer()
    output_path = Path(output_dir)

    rendered = renderer.render_note(note, output_path)

    print(f"✓ Rendered {len(rendered)} pages:")
    for path in rendered:
        print(f"  - {path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python saber_renderer.py <decrypted_file> <output_dir>")
        sys.exit(1)

    test_renderer(sys.argv[1], sys.argv[2])
