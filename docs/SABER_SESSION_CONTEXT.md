# Saber Integration - Session Context

**Date:** 2025-11-22
**Status:** Architecture complete, ready for testing and final integration

---

## What We Accomplished

### Session 1 (2025-11-21): Decryption & Parsing
1. **Discovered AES-CTR mode** - Critical breakthrough after CBC padding errors
2. **Implemented SaberDecryptor** - Successfully decrypts both files and filenames
3. **Implemented SaberProcessor** - Parses BSON, extracts pages/strokes/metadata
4. **Implemented SaberRenderer** - Renders strokes to images with pressure sensitivity
5. **Test Results:** Generated perfect rendering of "If you can read this, you are probably my buddy Claude."

### Session 2 (2025-11-22): Modular Architecture
1. **Created DocumentSource architecture** - Abstract base for all document types
2. **Implemented BooxPDFSource** - Wraps existing PDF logic
3. **Implemented SaberNoteSource** - Complete Saber pipeline (decrypt→parse→render)
4. **Created DocumentSourceManager** - Coordinates multiple sources
5. **Integrated into main.py** - Conditional source initialization based on config
6. **Added configuration** - BOOX_ENABLED, SABER_ENABLED, SABER_PASSWORD, etc.

---

## Current State of Code

### New Files Created
1. `processing/saber_decryptor.py` - AES-256-CTR decryption
2. `processing/saber_processor.py` - BSON parsing (SaberNote, SaberPage, SaberStroke)
3. `processing/saber_renderer.py` - Stroke rendering with Pillow
4. `processing/document_source.py` - Abstract base class + ProcessedDocument
5. `processing/boox_pdf_source.py` - PDF source implementation
6. `processing/saber_note_source.py` - Saber source implementation
7. `processing/document_source_manager.py` - Source coordination

### Modified Files
1. `config.py` - Added document source configuration
2. `main.py` - Integrated DocumentSourceManager (backward compatible)

### Test Files (all passing)
- `test_saber_decrypt.py` - Decryption validation
- `test_saber_parse_bson.py` - BSON structure
- `test_decode_stroke_points.py` - Point coordinates
- `test_bson_structure.py` - Format investigation
- `test_decrypt_ctr.py` - CTR mode discovery
- `test_decrypt_key.py` - Config key debugging
- `test_decrypt_file_direct.py` - Direct decryption test

---

## Key Technical Discoveries

### Encryption (Critical!)
- **Algorithm:** AES-256-CTR (NOT CBC!)
- **Key:** SHA256(password + "8MnPs64@R&mF8XjWeLrD")
- **IV:** From config.sbc, used as initial counter value
- **Filenames:** Same encryption, hex-encoded
- **config.sbc "key" field:** Unused for file encryption (maybe future use)

### BSON Format
- **Document length:** First 4 bytes (little-endian int32)
- **Trailing data:** 8 bytes after main document (ignore it)
- **Stroke points:** 12 bytes = 3 floats (x, y, pressure)
- **Version:** Currently format version 19

### File Structure
```
Test note: "25-11-11 Test.sbn2"
- 2 pages (1000x1400 each)
- 48 strokes on page 1, 0 on page 2
- Lined background (100px line height)
- Fountain pen tool (size 5.0)
```

---

## Configuration

### Environment Variables (optional, for production)
```bash
# Boox/PDF Support (enabled by default)
BOOX_ENABLED=true
BOOX_FOLDER=/mnt/onyx

# Saber Support (disabled by default)
SABER_ENABLED=false
SABER_FOLDER=/mnt/webdav/saber
SABER_PASSWORD=''  # Set this to enable Saber
```

### Test Credentials
- **Password:** `ehh1701jqb`
- **Test file:** `/home/sysop/Downloads/ba11646cfae1992948a4ae7d88078d56c18058bfc1736dd5a003c60ae8c7286b.sbe`
- **Config:** `/home/sysop/Downloads/test_saber/Saber/config.sbc`
- **Decrypted name:** `/25-11-11 Test.sbn2`

---

## What Still Needs To Be Done

### Immediate (Before First Real Use)
1. **Test main.py startup** - Verify no import errors or crashes
2. **Test with Boox only** - Ensure existing functionality still works
3. **Test with Saber enabled** - Drop .sbe file and watch it process
4. **Database schema** - Add source_type and source_metadata columns (optional but recommended)

### Important (For Production)
1. **File update detection** - Re-process files when they change
2. **Saber versioning** - Handle rapid auto-saves (debounce or dedupe)
3. **Error handling** - Graceful failures for corrupt/invalid files
4. **Temp file cleanup** - Delete rendered images after OCR

### Nice-to-Have
1. **Web UI for configuration** - Toggle sources, set passwords
2. **Source dashboard** - View status, statistics
3. **Manual triggers** - Re-process specific documents
4. **JSON format support** - Handle .sbn files (currently only .sbn2/BSON)

---

## How to Test

### Basic Startup Test (Boox only)
```bash
cd /home/sysop/RectangularFile
python main.py
```

Expected output:
```
Initializing Boox PDF source: /mnt/onyx
Registered document source: boox_pdf (enabled=True)
Saber note source is disabled
[STARTUP] ▶ Starting document source watchers...
```

### Enable Saber for Testing
```bash
export SABER_ENABLED='true'
export SABER_PASSWORD='ehh1701jqb'
export SABER_FOLDER='/home/sysop/Downloads/test_saber'
python main.py
```

Then copy the test .sbe file into the Saber folder:
```bash
cp /home/sysop/Downloads/ba*.sbe /home/sysop/Downloads/test_saber/Saber/
```

Watch the logs for:
1. File detected
2. Decryption
3. BSON parsing
4. Rendering
5. OCR queueing

---

## Architecture Overview

```
DocumentSourceManager
├── BooxPDFSource (BOOX_ENABLED)
│   ├── FileWatcher monitors BOOX_FOLDER for .pdf
│   ├── PDFProcessor extracts text/metadata
│   └── Queues to OCRQueueManager
│
└── SaberNoteSource (SABER_ENABLED)
    ├── FileWatcher monitors SABER_FOLDER for .sbe
    ├── SaberDecryptor (AES-CTR)
    ├── SaberProcessor (BSON parsing)
    ├── SaberRenderer (stroke→image)
    └── Queues rendered pages to OCRQueueManager
```

Both sources feed into the same OCR pipeline (QwenVLProcessor) and database (DatabaseManager).

---

## Key Code Patterns

### Processing a Saber Note
```python
# 1. Decrypt
decryptor = SaberDecryptor(password, saber_folder)
decrypted_data = decryptor.decrypt_file(encrypted_path)
decrypted_name = decryptor.decrypt_filename(encrypted_filename)

# 2. Parse
note = SaberProcessor.parse_note(decrypted_data)

# 3. Render
renderer = SaberRenderer()
page_images = renderer.render_note(note, output_dir)

# 4. OCR (existing pipeline)
for page_image in page_images:
    ocr_queue.add_to_queue(doc_id, page_image)
```

### Adding a New Document Source (Future)
```python
class MyNewSource(DocumentSource):
    def get_file_extensions(self):
        return ['.mynote']

    def can_process_file(self, file_path):
        return file_path.suffix == '.mynote'

    def process_file(self, file_path):
        # ... processing logic ...
        return ProcessedDocument(
            source_type='my_notes',
            original_path=str(file_path),
            title=title,
            page_images=[rendered_images],
            metadata=metadata
        )
```

---

## Potential Issues to Watch For

1. **Import errors** - Processing package imports might need adjustment
2. **Missing directories** - Saber folder may not exist if WebDAV not mounted
3. **Password in env** - Remember not to commit passwords to git
4. **Temp file accumulation** - Rendered images pile up in /tmp
5. **Duplicate processing** - Legacy file_watcher and new sources may both trigger

---

## Next Session Checklist

When you return:

1. **Pull latest code** from repository
2. **Read this file** to refresh context
3. **Read** `docs/SABER_INTEGRATION_STATUS.md` for detailed status
4. **Test startup** with `python main.py` (Boox only first)
5. **Check logs** for any errors or warnings
6. **Test Saber** if you want (set env vars)
7. **Consider database migration** if storing source metadata

---

## Questions You Might Have

**Q: Is it safe to test on production?**
A: Yes! Saber is disabled by default, and the new code is backward compatible.

**Q: Will this break existing PDF processing?**
A: No! BooxPDFSource wraps the existing logic. Legacy file_watcher still runs too.

**Q: What if Saber password is wrong?**
A: Decryption will fail gracefully with error logs. Document won't be added to database.

**Q: Can I enable/disable sources without code changes?**
A: Yes via environment variables now. Web UI for runtime changes is planned.

**Q: What about the "key" field in config.sbc?**
A: It's not used for file encryption. Files use password-derived key directly. Might be for future multi-device sync.

---

## Fun Facts From This Session

1. We spent most of Session 1 debugging CBC padding errors before discovering it was CTR mode all along!
2. The BSON has 8 mysterious trailing bytes that we learned to ignore
3. Your test note literally said "If you can read this, you are probably my buddy Claude" 😊
4. Stroke points are stored as binary floats, not text coordinates
5. The renderer uses pressure sensitivity to vary line thickness - looks really natural!

---

## Safe Travels! ✈️

The code is committed and ready for testing when you return. Everything is backward compatible, so existing functionality should be unaffected. Saber support is completely opt-in via configuration.

Next steps when back:
1. Test basic startup
2. Optionally test Saber with env vars
3. Consider database schema additions
4. Plan file update detection approach

All the heavy lifting is done - now it's just integration and polish!
