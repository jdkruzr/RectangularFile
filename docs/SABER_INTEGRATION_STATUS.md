# Saber Integration Status

**Last Updated:** 2025-11-22
**Status:** Phase 3 - Modular Architecture Implemented, Ready for Integration

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
SaberRenderer (✅ COMPLETE & TESTED)
    - Renders vector strokes to raster images
    - Applies pressure sensitivity to line width
    - Draws background patterns (lined, grid)
    - Outputs JPG for OCR pipeline
    ↓
DocumentSourceManager (✅ NEW ARCHITECTURE)
    - Coordinates multiple document sources
    - Manages file watchers for each source
    - Routes documents to OCR pipeline
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

3. **[processing/saber_renderer.py](../processing/saber_renderer.py)** ✅ COMPLETE & TESTED
   - `SaberRenderer` class
   - Renders strokes with pressure-sensitive line width
   - Draws lined/grid backgrounds
   - Outputs high-quality JPG for OCR
   - **Test render successful** - handwriting clearly legible

4. **[processing/document_source.py](../processing/document_source.py)** ✅ NEW ARCHITECTURE
   - Abstract `DocumentSource` base class
   - `ProcessedDocument` dataclass for unified output
   - Enables modular, pluggable document sources

5. **[processing/boox_pdf_source.py](../processing/boox_pdf_source.py)** ✅ COMPLETE
   - Wraps existing PDF processing logic
   - Implements DocumentSource interface
   - Independently toggleable via `BOOX_ENABLED`

6. **[processing/saber_note_source.py](../processing/saber_note_source.py)** ✅ COMPLETE
   - Full Saber note processing pipeline
   - Decrypt → Parse → Render → Output for OCR
   - Independently toggleable via `SABER_ENABLED`

7. **[processing/document_source_manager.py](../processing/document_source_manager.py)** ✅ COMPLETE
   - Coordinates multiple document sources
   - Manages FileWatcher instances per source
   - Routes processed documents to OCR pipeline

8. **Test Scripts** ✅ ALL PASSING
   - `test_saber_decrypt.py` - Decryption validation
   - `test_saber_parse_bson.py` - BSON structure exploration
   - `test_decode_stroke_points.py` - Point coordinate validation
   - `test_bson_structure.py` - Format investigation
   - `test_decrypt_ctr.py` - CTR mode discovery

### Modified Files
1. **[config.py](../config.py)** ✅ UPDATED
   - Added document source configuration section
   - `BOOX_ENABLED`, `BOOX_FOLDER` settings
   - `SABER_ENABLED`, `SABER_FOLDER`, `SABER_PASSWORD` settings

2. **[requirements.txt](../requirements.txt)**
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

## Current Status: Phase 3 Architecture Complete! 🎉

### Completed Sessions

#### Session 1 (2025-11-21): Core Saber Support
1. ✅ **Fixed Encryption** - Discovered files use CTR mode, not CBC
2. ✅ **Successful Decryption** - Both filenames and file contents decrypt perfectly
3. ✅ **BSON Parsing** - Complete note structure extracted
4. ✅ **Point Decoding** - All stroke coordinates decoded (x, y, pressure)
5. ✅ **Renderer Implementation** - Stroke rendering with pressure sensitivity
6. ✅ **Renderer Testing** - Generated perfect image: "If you can read this, you are probably my buddy Claude."

#### Session 2 (2025-11-22): Modular Architecture
1. ✅ **Abstract DocumentSource** - Base class for all document sources
2. ✅ **BooxPDFSource** - Modular PDF support (existing functionality)
3. ✅ **SaberNoteSource** - Complete Saber pipeline as pluggable source
4. ✅ **DocumentSourceManager** - Coordinates multiple sources
5. ✅ **Configuration** - Per-source enable/disable flags

### Test Results
- **Test file:** `25-11-11 Test.sbn2` (2 pages, 48 strokes, lined paper)
- **Decryption:** ✅ Success (147,744 bytes)
- **BSON parsing:** ✅ Success (version 19 format)
- **Stroke points:** ✅ All decoded (61 points in first stroke)
- **Renderer:** ✅ PERFECT - Handwriting clearly legible in generated JPEG

---

## Phase 4: Remaining Integration Tasks

### 1. Update main.py ⏳ NEXT
- Replace single FileWatcher with DocumentSourceManager
- Initialize BooxPDFSource and SaberNoteSource
- Wire up to existing OCR queue
- Test with both source types enabled

### 2. Database Schema Updates ⏳ REQUIRED
Add to `pdf_documents` table:
- `source_type` column: VARCHAR ('boox_pdf', 'saber_note', 'html')
- `source_metadata` column: JSON (for source-specific data)
- Migration script to add columns to existing database

Example metadata per source:
- **Boox:** `{'has_embedded_text': true, 'file_size': 12345}`
- **Saber:** `{'background_pattern': 'lined', 'total_strokes': 48, 'encrypted_filename': '...'}`

### 3. OCR Pipeline Updates ⏳ MINOR
- Handle ProcessedDocument objects (already have page_images)
- Store source_type and metadata in database
- Support multi-page documents from Saber (iterate page_images)

### 4. File Update Detection ⏳ IMPORTANT
**Current Issue:** Both Boox and Saber can update existing files, but we only process on creation.

**Proposed Solution:**
- Track file modification times in database
- FileWatcher already detects mtime changes
- Add "update" callback separate from "new file" callback
- Re-run OCR when file mtime increases
- Store version history (optional)

**Saber-specific:** Saber may create multiple versions of same note (one per edit). Need logic to:
- Identify canonical version (latest by mtime)
- Avoid processing every intermediate save
- Possibly debounce updates (wait for file to stabilize)

### 5. Web UI Configuration Page ⏳ NICE-TO-HAVE
Add settings page for managing document sources:
- Toggle Boox/Saber sources on/off
- Configure folders for each source
- Set Saber encryption password
- Test connection/decryption
- View source status and statistics

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

## Open Questions & Considerations

### Resolved ✅
1. ✅ **How are files encrypted?** - AES-CTR with password-derived key
2. ✅ **Do we need config.sbc?** - Yes, for the IV (used as counter)
3. ✅ **What metadata is in files?** - Complete: version, pages, strokes, background pattern, etc.
4. ✅ **How to render strokes?** - Pillow with pressure-sensitive line width
5. ✅ **Does rendering quality work for OCR?** - YES! Perfectly legible

### To Be Addressed ⏳
6. **File update detection** - How to handle when files are modified after initial import?
   - Boox: Manual PDF uploads may be replaced/updated
   - Saber: Frequent auto-saves create multiple versions
   - Need: mtime tracking + re-processing logic

7. **Saber versioning** - How to handle rapid successive saves?
   - Option A: Process every version (lots of OCR overhead)
   - Option B: Debounce and only process stable files (e.g., no changes for 30s)
   - Option C: Only process latest version per note title

8. **Performance** - Will Saber rendering + OCR be fast enough?
   - Need to test with real multi-page notes
   - May need async rendering or queue management

---

## Todo List

### Completed ✅
- [x] Research Saber file format
- [x] Understand encryption scheme (AES-CTR discovery!)
- [x] Create SaberDecryptor class
- [x] Create comprehensive test scripts
- [x] Run all decryption tests - ALL PASSING
- [x] Create SaberProcessor for BSON parsing
- [x] Implement stroke rendering (SaberRenderer)
- [x] Test renderer output quality - PERFECT
- [x] Design modular architecture
- [x] Create DocumentSource base class
- [x] Create BooxPDFSource and SaberNoteSource
- [x] Create DocumentSourceManager
- [x] Add configuration options to config.py

### Remaining Tasks ⏳

#### High Priority (Core Functionality)
- [ ] **Update main.py** - Integrate DocumentSourceManager
- [ ] **Database schema migration** - Add source_type and source_metadata columns
- [ ] **OCR pipeline integration** - Handle ProcessedDocument format
- [ ] **File update detection** - Track mtime changes and re-process
- [ ] **End-to-end testing** - Process real Saber note through full pipeline

#### Medium Priority (Robustness)
- [ ] Saber file versioning/deduplication logic
- [ ] Error handling for corrupt/invalid Saber files
- [ ] Temp file cleanup on shutdown
- [ ] Logging improvements

#### Low Priority (Polish)
- [ ] Web UI for source configuration
- [ ] Source status dashboard
- [ ] Manual source enable/disable without restart
- [ ] Support for .sbn (JSON) format in addition to .sbn2 (BSON)
