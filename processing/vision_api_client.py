"""
OpenAI-compatible Vision API client for document OCR.

This module provides a client that speaks the OpenAI Chat Completions API format,
allowing RectangularFile to work with any compatible backend:
- vLLM (local)
- Ollama (local)
- llama.cpp / llama-server (local)
- OpenAI GPT-4o (cloud)
- Anthropic Claude via proxy (cloud)
- Any other OpenAI-compatible endpoint
"""
import base64
import json
import logging
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """Base class for inference errors."""
    pass


class APIConnectionError(InferenceError):
    """Cannot reach the inference API."""
    pass


class APIResponseError(InferenceError):
    """API returned an error response."""
    pass


class VisionAPIClient:
    """
    Client for OpenAI-compatible vision APIs.

    Handles document OCR by converting PDFs to images and sending them
    to a vision-language model for transcription and annotation detection.
    """

    # Prompts for different tasks
    TRANSCRIPTION_PROMPT = """Transcribe all handwritten text in this image exactly as written.
Preserve line breaks, paragraph structure, and formatting as closely as possible.
If text is unclear, make your best attempt and indicate uncertainty with [?].
Output only the transcription, no commentary."""

    ANNOTATION_PROMPT = """Analyze this handwritten note image for colored annotations.
Look for:
1. RED ink/pen marks - these indicate TODOs or action items
2. GREEN ink/pen marks - these indicate tags or categories

For each colored annotation you find, extract the text and output a JSON object on its own line:
{"type": "todo", "text": "the red text content"}
{"type": "tag", "text": "the green text content"}

Only output JSON lines for annotations you actually find. If there are no colored annotations, output nothing.
Do not include any other text or explanation."""

    def __init__(
        self,
        api_base: str,
        api_key: str = "",
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_tokens: int = 2048,
        timeout: int = 120,
    ):
        """
        Initialize the Vision API client.

        Args:
            api_base: Base URL for the API (e.g., "http://localhost:8000/v1")
            api_key: API key (optional for local servers)
            model: Model identifier to use
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        """
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

        logger.info(f"VisionAPIClient initialized: {self.api_base}, model={self.model}")

    def _image_to_base64(self, image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
        """
        Convert PIL Image to base64 data URL.

        Args:
            image: PIL Image to convert
            format: Image format (JPEG or PNG)
            quality: JPEG quality (ignored for PNG)

        Returns:
            Data URL string (data:image/jpeg;base64,...)
        """
        buffer = io.BytesIO()

        if format.upper() == "JPEG":
            # Convert to RGB if necessary (JPEG doesn't support alpha)
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            image.save(buffer, format="JPEG", quality=quality)
            mime = "image/jpeg"
        else:
            image.save(buffer, format="PNG")
            mime = "image/png"

        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:{mime};base64,{b64}"

    def _build_headers(self) -> dict:
        """Build request headers including auth if configured."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _resize_if_needed(self, image: Image.Image, max_dimension: int = 2048) -> Image.Image:
        """
        Resize image if it exceeds max dimensions.

        Many vision APIs have size limits. This ensures we don't exceed them
        while maintaining aspect ratio.

        Args:
            image: PIL Image
            max_dimension: Maximum width or height

        Returns:
            Resized image (or original if already within limits)
        """
        if max(image.size) <= max_dimension:
            return image

        ratio = max_dimension / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        logger.debug(f"Resizing image from {image.size} to {new_size}")
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def _call_api(self, prompt: str, image: Image.Image) -> str:
        """
        Make an API call with an image and prompt.

        Args:
            prompt: Text prompt for the model
            image: PIL Image to analyze

        Returns:
            Model's text response

        Raises:
            APIConnectionError: Cannot reach the API
            APIResponseError: API returned an error
        """
        image = self._resize_if_needed(image)
        image_url = self._image_to_base64(image)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "max_tokens": self.max_tokens,
        }

        try:
            response = self.client.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                headers=self._build_headers(),
            )
        except httpx.ConnectError as e:
            raise APIConnectionError(f"Cannot connect to API at {self.api_base}: {e}")
        except httpx.TimeoutException as e:
            raise APIConnectionError(f"API request timed out after {self.timeout}s: {e}")

        if response.status_code != 200:
            raise APIResponseError(
                f"API returned status {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise APIResponseError(f"Unexpected API response format: {e}")

    def transcribe_image(self, image: Image.Image, prompt: str = None) -> str:
        """
        Transcribe handwritten text from an image.

        Args:
            image: PIL Image containing handwritten text
            prompt: Optional custom prompt (uses default if not provided)

        Returns:
            Transcribed text
        """
        if prompt is None:
            prompt = self.TRANSCRIPTION_PROMPT

        return self._call_api(prompt, image)

    def detect_annotations(self, image: Image.Image) -> list[dict]:
        """
        Detect colored annotations (red TODOs, green tags) in an image.

        Args:
            image: PIL Image to analyze

        Returns:
            List of annotation dictionaries with 'type' and 'text' keys
        """
        response_text = self._call_api(self.ANNOTATION_PROMPT, image)

        annotations = []
        for line in response_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            # Try to parse as JSON
            if line.startswith('{'):
                try:
                    annotation = json.loads(line)
                    # Normalize type names
                    ann_type = annotation.get('type', '').lower()
                    if ann_type in ('todo', 'red', 'action'):
                        annotation['type'] = 'todo'
                    elif ann_type in ('tag', 'green', 'category'):
                        annotation['type'] = 'tag'
                    annotations.append(annotation)
                except json.JSONDecodeError:
                    logger.debug(f"Could not parse annotation line: {line}")
                    continue

        return annotations

    def process_pdf_page(
        self,
        pdf_path: Path,
        page_number: int,
        dpi: int = 150
    ) -> dict:
        """
        Process a single PDF page: transcribe and detect annotations.

        Args:
            pdf_path: Path to the PDF file
            page_number: 1-indexed page number
            dpi: Resolution for PDF rendering

        Returns:
            Dictionary with 'text' and 'annotations' keys
        """
        images = convert_from_path(
            pdf_path,
            first_page=page_number,
            last_page=page_number,
            dpi=dpi
        )

        if not images:
            logger.warning(f"No image generated for page {page_number} of {pdf_path}")
            return {"text": "", "annotations": []}

        image = images[0]

        # Transcribe the page
        text = self.transcribe_image(image)

        # Detect annotations
        annotations = self.detect_annotations(image)

        return {
            "text": text,
            "annotations": annotations,
        }

    def process_document(
        self,
        filepath: Path,
        doc_id: int,
        db_manager,
        progress_callback=None
    ) -> bool:
        """
        Process an entire document (PDF) through OCR.

        This is the main entry point that replaces QwenVLProcessor.process_document().

        Args:
            filepath: Path to the PDF file
            doc_id: Document ID in the database
            db_manager: Database manager for storing results
            progress_callback: Optional callback for progress updates

        Returns:
            True if processing succeeded, False otherwise
        """
        logger.info(f"Processing document {doc_id}: {filepath}")

        try:
            # Get page count
            from pdf2image.pdf2image import pdfinfo_from_path
            pdf_info = pdfinfo_from_path(str(filepath))
            page_count = pdf_info.get('Pages', 1)

            logger.info(f"Document has {page_count} pages")
            db_manager.mark_ocr_started(doc_id)

            page_data = {}
            all_annotations = []

            for page_num in range(1, page_count + 1):
                progress = (page_num / page_count) * 100
                db_manager.update_processing_progress(
                    doc_id, progress, f"Processing page {page_num}/{page_count}"
                )

                logger.info(f"Processing page {page_num}/{page_count}")

                try:
                    result = self.process_pdf_page(filepath, page_num)

                    page_data[page_num] = {
                        'text': result['text'],
                        'word_count': len(result['text'].split()),
                        'processed_at': datetime.now(),
                    }

                    # Collect annotations with page numbers
                    for ann in result['annotations']:
                        all_annotations.append({
                            'page_number': page_num,
                            'annotation_type': ann.get('type', 'unknown'),
                            'text': ann.get('text', ''),
                            'confidence': 1.0,  # API doesn't provide confidence
                        })

                except Exception as page_error:
                    logger.error(f"Error processing page {page_num}: {page_error}")
                    page_data[page_num] = {
                        'text': f"[Error processing page: {page_error}]",
                        'word_count': 0,
                        'processed_at': datetime.now(),
                    }

            # Store results
            db_manager.store_ocr_results(doc_id, page_data)

            if all_annotations:
                db_manager.store_document_annotations(doc_id, all_annotations)
                logger.info(f"Stored {len(all_annotations)} annotations for document {doc_id}")

            logger.info(f"Document {doc_id} processing complete")
            return True

        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Mark as failed in database
            db_manager.update_processing_progress(
                doc_id, 0, f"Processing failed: {e}"
            )
            return False

    def health_check(self) -> dict:
        """
        Check if the API endpoint is reachable and responding.

        Returns:
            Dictionary with 'healthy' boolean and optional 'error' or 'models' info
        """
        try:
            response = self.client.get(
                f"{self.api_base}/models",
                headers=self._build_headers(),
                timeout=5,
            )

            if response.status_code == 200:
                data = response.json()
                models = [m.get('id', 'unknown') for m in data.get('data', [])]
                return {
                    "healthy": True,
                    "models": models,
                }
            else:
                return {
                    "healthy": False,
                    "error": f"Status {response.status_code}",
                }

        except httpx.ConnectError as e:
            return {
                "healthy": False,
                "error": f"Connection failed: {e}",
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
            }

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
