# Saber Integration Status

**Last Updated:** 2025-11-21
**Status:** Phase 2 - BSON Parsing & Rendering Complete, Testing Renderer

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
SaberDecryptor (✅ COMPLETE)
    - Reads config.sbc for IV
    - Derives key from user password (AES-256-CTR)
    - Decrypts .sbe files → raw BSON
    - Decrypts filenames (hex-encoded)
    ↓
SaberProcessor (✅ COMPLETE)
    - Parses BSON note structure
    - Extracts pages, strokes, metadata
    - Decodes stroke points (x, y, pressure)
    ↓
SaberRenderer (✅ IMPLEMENTED, ⏳ TESTING)
    - Renders vector strokes to raster images
    - Applies pressure sensitivity to line width
    - Draws background patterns (lined, grid)
    - Outputs JPG for OCR pipeline
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

### Encryption Scheme ✅ VALIDATED
1. **Key Derivation:** `SHA256(password + "8MnPs64@R&mF8XjWeLrD")`
2. **Algorithm:** AES-256-CTR (Counter mode - no padding!)
3. **IV:** Stored in `config.sbc` as base64, used as initial counter value
4. **Filenames:** Encrypted with same key, hex-encoded

**Key Discovery:** Files use **CTR mode**, not CBC! This explains why CBC attempts failed with padding errors.

### config.sbc Structure
```json
{
  "key": "40cFFoKmjCwNwM8ptjzFbeuaMouYMVON53iMUbaBbD7mgaShLiFrR4eZIGwRbgjx",
  "iv": "SC7h0SRsA5IN8QBhgBURfA=="
}
```

**Note:** The `"key"` field appears to be for future use or multi-device sync. Files are encrypted **directly** with the password-derived key, not this stored key.

---

## Files Created/Modified

### New Files
1. **[processing/saber_decryptor.py](../processing/saber_decryptor.py)** ✅ COMPLETE
   - `SaberDecryptor` class
   - AES-256-CTR decryption with password-derived key
   - Decrypts both .sbe files and hex-encoded filenames
   - Successfully tested with real Saber notes

2. **[processing/saber_processor.py](../processing/saber_processor.py)** ✅ COMPLETE
   - `SaberNote`, `SaberPage`, `SaberStroke` classes
   - Parses BSON structure (handles trailing bytes correctly)
   - Decodes binary point data (x, y, pressure as floats)
   - Extracts all metadata (version, background pattern, etc.)

3. **[processing/saber_renderer.py](../processing/saber_renderer.py)** ✅ IMPLEMENTED
   - `SaberRenderer` class
   - Renders strokes with pressure-sensitive line width
   - Draws lined/grid backgrounds
   - Outputs high-quality JPG for OCR

4. **Test Scripts** ✅ ALL PASSING
   - `test_saber_decrypt.py` - Decryption validation
   - `test_saber_parse_bson.py` - BSON structure exploration
   - `test_decode_stroke_points.py` - Point coordinate validation
   - `test_bson_structure.py` - Format investigation
   - `test_decrypt_ctr.py` - CTR mode discovery

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

## Current Status: Phase 2 Complete! 🎉

### Completed Today (2025-11-21)
1. ✅ **Fixed Encryption** - Discovered files use CTR mode, not CBC
2. ✅ **Successful Decryption** - Both filenames and file contents decrypt perfectly
3. ✅ **BSON Parsing** - Complete note structure extracted
4. ✅ **Point Decoding** - All stroke coordinates decoded (x, y, pressure)
5. ✅ **Renderer Implementation** - Stroke rendering with pressure sensitivity

### Test Results
- **Test file:** `25-11-11 Test.sbn2` (2 pages, 48 strokes, lined paper)
- **Decryption:** ✅ Success (147,744 bytes)
- **BSON parsing:** ✅ Success (version 19 format)
- **Stroke points:** ✅ All decoded (61 points in first stroke)
- **Renderer:** ⏳ Ready to test

---

## Phase 3: Next Steps (Integration & Testing)

### 1. Test Stroke Rendering
```bash
python3 processing/saber_renderer.py <decrypted_file> <output_dir>
```
- Verify rendered images look correct
- Check pressure sensitivity rendering
- Validate lined background appearance

### 2. FileWatcher Integration
- Add `.sbe` file detection to `file_watcher.py`
- Implement file versioning logic:
  - Saber saves frequently (possibly one file per change)
  - Need to identify canonical version using filename + timestamps
  - Handle updates when files change

### 3. Configuration
Add to [config.py](../config.py):
```python
# Saber integration settings
SABER_FOLDER = os.getenv('SABER_FOLDER', '/mnt/webdav/saber')
SABER_ENC_PASSWORD = os.getenv('SABER_ENC_PASSWORD', '')
SABER_ENABLED = os.getenv('SABER_ENABLED', 'false').lower() == 'true'
```

### 4. Database Schema Updates
Track source type in `pdf_documents` table:
- Add `source_type` column: 'boox_pdf', 'saber_note', 'html'
- Store original encrypted filename
- Store decrypted path/title
- Store Saber metadata (background pattern, tool types used, etc.)

---

## Key Decisions Made

### Decryption Approach ✅ VALIDATED
- **AES-256-CTR mode** (not CBC!) - critical discovery via testing
- Files encrypted directly with password-derived key
- The "key" field in config.sbc is unused for file decryption
- Filenames decrypted the same way, stored as hex strings
- Config.sbc provides the IV (used as initial counter value)

### BSON Parsing Strategy ✅ IMPLEMENTED
- Files have 8 trailing bytes after main BSON document
- Use document length from first 4 bytes to parse correctly
- Stroke points stored as binary: 12 bytes = 3 floats (x, y, pressure)
- All metadata successfully extracted and validated

### Rendering Strategy ✅ IMPLEMENTED
- Convert vector strokes to raster images using Pillow
- Apply pressure sensitivity to line width for realistic appearance
- Draw background patterns (lined/grid) beneath strokes
- Output as JPEG for compatibility with existing OCR pipeline
- No changes needed to QwenVL OCR code

### Versioning Strategy (To Be Implemented)
- Decrypt filename to get original path
- Use filename + modification time to detect updates
- Treat as single document with version history
- Re-run OCR when file changes

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

1. ✅ **How are files encrypted?** - Solved: AES-CTR with password-derived key
2. ✅ **Do we need config.sbc?** - Yes, for the IV (used as counter)
3. ✅ **What metadata is in files?** - Complete: version, pages, strokes, background pattern, etc.
4. ✅ **How to render strokes?** - Pillow with pressure-sensitive line width
5. ⏳ **How to handle rapid updates?** - Need to implement versioning logic
6. ⏳ **Does rendering quality work for OCR?** - Need to test with QwenVL

---

## Todo List

- [x] Research Saber file format
- [x] Understand encryption scheme
- [x] Create SaberDecryptor class (AES-CTR)
- [x] Create test scripts
- [x] Run decryption tests - ALL PASSING
- [x] Create SaberProcessor for BSON parsing
- [x] Implement stroke rendering (SaberRenderer)
- [ ] **CURRENT:** Test renderer output quality
- [ ] Integrate with FileWatcher
- [ ] Add configuration options
- [ ] Connect to QwenVL OCR pipeline
- [ ] Test with live WebDAV sync
