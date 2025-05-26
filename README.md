# RectangularFile
Rectangular File is a document management system originally intended for use with Onyx Boox e-ink tablets, to enhance your ability to search for text in your handwriting and build an index of recognized text for later use. Supernote does this with their handwriting search functionality, but Boox devices start over every time you start a new search. This, how you say, Le Sucks.

RF is a Flask application which uses Tesseract OCR with handwriting-specific models that you can train to be more accurate for your handwriting style. You deploy it on a server somewhere, tell your tablet devices to dump their data into a folder on the server -- on Boox this is just an auto-PDF-upload feature -- and RF will ingest them, perform text extraction and handwriting recognition, and then add the data to its searchable index, then show you where it got the search results from when it finds what you're looking for.

Eventually we'd also like to add features like wordclouds, etc.

You could conceivably use this with notes from any device -- Kindle Scribe, reMarkable, etc.

## System Dependencies

Before installing the Python requirements, you need to install these system packages:

### macOS (using Homebrew):
```bash
brew install tesseract
brew install tesseract-lang  # Optional language packs
brew install poppler  # Required for PDF processing
```

### Debian/Ubuntu:
```bash
apt-get update
apt-get install -y tesseract-ocr
apt-get install -y tesseract-ocr-eng  # English language pack
apt-get install -y poppler-utils  # Required for PDF processing
```

### Windows:
Not supported.
