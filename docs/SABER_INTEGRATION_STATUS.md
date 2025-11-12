# Saber Integration Status

**Last Updated:** 2025-11-12
**Status:** Phase 1 - Decryption Implementation Complete, Testing in Progress

---

## Overview

Integrating Saber note-taking app support into RectangularFile to create a unified personal knowledge management system that works with:
- **Boox tablets** → Native .note files (existing)
- **LCD devices** (Galaxy S23 Ultra, Wacom Movinkpad 11) → Saber notes (new)

Both sync via WebDAV to a common backend.

---

## Current Architecture

```
Saber Device (Galaxy S23 / Wacom)
    ↓ writes encrypted .sbe files
WebDAV Server (watched folder)
    ↓ RF FileWatcher detects .sbe files + config.sbc
SaberDecryptor (NEW - IMPLEMENTED)
    - Reads config.sbc for IV
    - Derives key from user password
    - Decrypts .sbe files → raw BSON
    ↓
SaberProcessor (TODO - NEXT STEP)
    - Parses BSON note structure
    - Extracts pages, strokes, images
    - Renders each page to raster image
    ↓
QwenVLProcessor (existing OCR)
    - Performs handwriting recognition
    ↓
DatabaseManager (existing)
    - Stores text, metadata, annotations
```

---

## Saber Encryption Details (VALIDATED)

### File Format
- Encrypted files: `.sbe` extension
- Unencrypted files: `.sbn2` (BSON) or `.sbn` (JSON)
- Metadata file: `config.sbc` (unencrypted JSON)

### Encryption Scheme
1. **Key Derivation:** `SHA256(password + "8MnPs64@R&mF8XjWeLrD")`
2. **Algorithm:** AES-256-CBC
3. **IV:** Stored in `config.sbc` as base64
4. **Filenames:** Also encrypted (hex-encoded)

### config.sbc Structure
```json
{
  "key": "40cFFoKmjCwNwM8ptjzFbeuaMouYMVON53iMUbaBbD7mgaShLiFrR4eZIGwRbgjx",
  "iv": "SC7h0SRsA5IN8QBhgBURfA=="
}
```

Note: The `"key"` field is the file encryption key encrypted with the password-derived key. Files are encrypted directly with the password-derived key using the IV from config.

---

## Files Created/Modified

### New Files
1. **[processing/saber_decryptor.py](../processing/saber_decryptor.py)**
   - `SaberDecryptor` class
   - Implements password → key derivation
   - Decrypts .sbe files and filenames
   - Status: ✅ IMPLEMENTED, ⏳ TESTING

2. **[test_saber_decrypt.py](../test_saber_decrypt.py)**
   - Test script for decryption
   - Tests both filename and file content decryption
   - Status: ✅ CREATED, ⏳ READY TO RUN

3. **Test Data**
   - `/home/jtd/Downloads/test_saber/Saber/config.sbc` - Config file for testing
   - `/home/jtd/Downloads/ba11646cfae1992948a4ae7d88078d56c18058bfc1736dd5a003c60ae8c7286b.sbe` - Test note

### Modified Files
1. **[requirements.txt](../requirements.txt)**
   - Added: `pycryptodome>=3.18.0` (for AES decryption)
   - Added: `pymongo>=4.0.0` (for BSON parsing)

---

## Test Credentials

### Test File Details
- **Encrypted filename:** `ba11646cfae1992948a4ae7d88078d56c18058bfc1736dd5a003c60ae8c7286b.sbe`
- **Original filename:** `25-11-11 Test.sbn2`
- **Original path:** `/25-11-11 Test.sbn2`
- **Saber encryption password:** `ehh1701jqb`

### Test Setup
```bash
# Install dependencies
pip install pycryptodome pymongo

# Run decryption test
python3 test_saber_decrypt.py
```

---

## Current Status: BLOCKED

### Blocking Issue
Waiting for `pycryptodome` to be installed on the system.

### Once Unblocked
1. Run `test_saber_decrypt.py` to verify decryption works
2. If successful, proceed to Phase 2 (BSON parsing)
3. If fails, debug encryption implementation

---

## Phase 2: Next Steps (After Decryption Test Passes)

### 1. Create SaberProcessor
- Parse BSON structure using `pymongo`/`bson` library
- Extract note metadata (version, pages, background)
- Handle both `.sbn2` (BSON) and `.sbn` (JSON) formats

### 2. Implement Stroke Rendering
- File: `processing/saber_renderer.py`
- Render vector strokes to raster images
- Use Pillow or cairo for drawing
- Input: Page dimensions, stroke points (x, y, pressure)
- Output: JPG/PNG for OCR pipeline

### 3. FileWatcher Integration
- Add `.sbe` file detection to `file_watcher.py`
- Implement file versioning logic:
  - Saber saves frequently (possibly one file per change)
  - Need to identify canonical version using filename + timestamps
  - Handle updates when files change

### 4. Configuration
Add to [config.py](../config.py):
```python
# Saber integration settings
SABER_FOLDER = os.getenv('SABER_FOLDER', '/mnt/webdav/saber')
SABER_ENC_PASSWORD = os.getenv('SABER_ENC_PASSWORD', '')
SABER_ENABLED = os.getenv('SABER_ENABLED', 'false').lower() == 'true'
```

### 5. Database Schema Updates
Track source type in `pdf_documents` table:
- Add `source_type` column: 'boox_pdf', 'saber_note', 'html'
- Store original encrypted filename
- Store decrypted path/title

---

## Key Decisions Made

### Decryption Approach
- Files encrypted directly with password-derived key (not the key in config.sbc)
- Must decrypt filenames to track note identity/versions
- Config.sbc provides the IV needed for decryption

### Versioning Strategy (To Be Implemented)
- Decrypt filename to get original path
- Use filename + modification time to detect updates
- Treat as single document with version history
- Re-run OCR when file changes

### Rendering Strategy (To Be Implemented)
- Convert vector strokes to raster images
- Feed to existing Qwen OCR pipeline
- No changes needed to OCR code

---

## Saber File Format Reference

### BSON Structure (from Saber source)
```dart
{
  'v': sbnVersion,              // File format version (currently 19)
  'ni': nextImageId,            // Next image ID counter
  'b': backgroundColor,         // Background color (ARGB32)
  'p': backgroundPattern.name,  // Pattern name (string)
  'l': lineHeight,              // Line height (int)
  'lt': lineThickness,          // Line thickness (int)
  'z': [pages],                 // Array of EditorPage objects
  'c': initialPageIndex,        // Current page index
}
```

### EditorPage Structure
Each page contains:
- `width`, `height`: Page dimensions
- `strokes`: Array of Stroke objects (vector paths)
- `images`: Array of EditorImage objects

### Stroke Structure
- Points array (x, y, pressure)
- Color
- Tool type (pen, pencil, highlighter, etc.)
- Stroke properties

---

## Reference Files in Saber Source

Key files examined:
- `lib/data/editor/editor_core_info.dart` - File format/BSON structure
- `lib/data/nextcloud/nextcloud_client_extension.dart` - Encryption scheme
- `lib/data/nextcloud/saber_syncer.dart` - File sync and encryption
- `lib/data/file_manager/file_manager.dart` - File handling
- `lib/pages/editor/editor.dart` - Editor constants

---

## Contact & Context Transfer

When resuming this work on another machine:

1. **Pull latest code** from git
2. **Read this document** to understand current state
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run test:**
   ```bash
   python3 test_saber_decrypt.py
   ```
5. **If test passes:** Move to Phase 2 (SaberProcessor)
6. **If test fails:** Debug decryption, check password/config.sbc

---

## Open Questions

1. ✅ **How are files encrypted?** - Solved: AES-CBC with password-derived key
2. ✅ **Do we need config.sbc?** - Yes, for the IV
3. ⏳ **How to handle rapid updates?** - Need to implement versioning logic
4. ⏳ **What metadata is in files?** - Need to parse BSON to confirm
5. ⏳ **How to render strokes?** - Will use Pillow/cairo (TBD in Phase 2)

---

## Todo List

- [x] Research Saber file format
- [x] Understand encryption scheme
- [x] Create SaberDecryptor class
- [x] Create test script
- [ ] **CURRENT:** Run decryption test (blocked on pycryptodome)
- [ ] Create SaberProcessor for BSON parsing
- [ ] Implement stroke rendering
- [ ] Integrate with FileWatcher
- [ ] Add configuration options
- [ ] Test with live WebDAV sync
