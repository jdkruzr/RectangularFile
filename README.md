# RectangularFile

![image](https://github.com/user-attachments/assets/c168eea6-1992-473e-9427-39dbf52b12c9)

RectangularFile is a powerful open-source document management system designed for handwritten notes from e-ink tablets (Onyx Boox, reMarkable, Supernote, etc.) or any kind of device that can output PDFs of handwritten content, such as iPads or Samsung tablets. It uses any OpenAI-compatible vision API — local (vLLM, Ollama) or cloud (OpenAI, Claude) — to transcribe handwriting, detect annotations, and make everything searchable.

## ✨ Features

- **🤖 AI-Powered Handwriting Recognition** - Transcribes handwritten notes via any OpenAI-compatible vision API
- **📝 Annotation Detection** - Automatically detects and indexes:
  - 🔴 Red ink → TODOs and action items
  - 🟢 Green ink → Tags and categories
- **📅 CalDAV To-Do Sync** - Automatically creates todos on your CalDAV server from red-ink action items
- **🔍 Full-Text Search** - Search across all your handwritten notes with folder filtering
- **📁 Multi-Device Support** - Automatically organizes notes from multiple devices
- **☁️ Word Clouds** - Visualize common themes across your notes
- **✏️ In-Place Editing** - Fix transcription errors directly in the web interface
- **🔒 Simple Authentication** - Single-user login system

## 🚀 Quick Start

1. Clone the repository
2. Create a virtualenv in your repo directory: `python3 -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Set up authentication (see below)
5. Configure your inference API endpoint (see below)
6. Set up a systemd unit file using the provided [rectangular-file.service](rectangular-file.service) example, place in `/etc/systemd/system/`
7. Point your e-ink devices to sync PDFs to `/mnt/onyx` (or your configured folder)
8. `systemctl start && systemctl enable rectangular-file`

## 📋 Requirements

- Python 3.8+
- An OpenAI-compatible vision API endpoint (local or cloud — see [Inference API Setup](#inference-api-setup))
- System packages:
  - `poppler-utils` (Ubuntu/Debian: `sudo apt install poppler-utils`, macOS: `brew install poppler`)
  - `python3-pip python3-venv` (Linux)

## 🔐 Authentication Setup

Generate a secure secret key:
```bash
python -c 'import secrets; print(secrets.token_hex(32))'
```

Generate password hash:
```bash
python -c "import hashlib; print(hashlib.sha256('yourpassword'.encode()).hexdigest())"
```

Add to your systemd service or environment:

```bash
SECRET_KEY=<generated_key>
APP_PASSWORD_HASH=<generated_hash>
```

## 🧠 Inference API Setup

RectangularFile works with any OpenAI-compatible vision API. Configure it with these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_API_BASE` | `http://localhost:8000/v1` | OpenAI-compatible API endpoint |
| `INFERENCE_API_KEY` | *(empty)* | API key (required for cloud providers) |
| `INFERENCE_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | Model identifier |
| `INFERENCE_MAX_TOKENS` | `2048` | Max tokens per response |
| `INFERENCE_TIMEOUT` | `120` | Request timeout in seconds |

**Local backends:**
- **vLLM**: `INFERENCE_API_BASE=http://localhost:8000/v1` (default)
- **Ollama**: `INFERENCE_API_BASE=http://localhost:11434/v1`
- **llama.cpp**: `INFERENCE_API_BASE=http://localhost:8080/v1`

**Cloud backends:**
- **OpenAI**: `INFERENCE_API_BASE=https://api.openai.com/v1` with `INFERENCE_API_KEY=sk-...` and `INFERENCE_MODEL=gpt-4o`
- **Anthropic** (via OpenAI-compatible proxy): set `INFERENCE_API_BASE` to your proxy URL

A vision-capable model is required. [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) works well for handwriting; GPT-4o and similar cloud models also work.

## ⚙️ Configuration

All settings are configured via environment variables. See [rectangular-file.service](rectangular-file.service) for a complete example.

### Key Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `UPLOAD_FOLDER` | `/mnt/onyx` | Directory where PDFs are synced |
| `DATABASE_PATH` | `/mnt/rectangularfile/pdf_index.db` | SQLite database location |
| `ARCHIVE_ENABLED` | `true` | Move processed files to archive (enables re-upload detection) |
| `ARCHIVE_FOLDER` | `/mnt/rectangularfile/archive` | Archive directory for processed files |
| `DEBUG_IMAGES_DIR` | `/mnt/rectangularfile/debug_images` | Debug image output |
| `POLLING_INTERVAL` | `30.0` | File watcher polling interval (seconds) |
| `FLASK_HOST` | `0.0.0.0` | Flask server bind address |
| `FLASK_PORT` | `5000` | Flask server port |

### How Archiving Works

When a document is successfully processed, it is moved from `UPLOAD_FOLDER` to `ARCHIVE_FOLDER`. This keeps the watch directory clean and enables re-upload detection: if a file with the same name appears in the upload folder again, the system treats it as new content.

If you disable archiving (`ARCHIVE_ENABLED=false`), files remain in the upload folder after processing.

### Starting from Scratch

To rebuild your database from existing files:

1. **Stop the service:**
   ```bash
   sudo systemctl stop rectangular-file
   ```

2. **Backup your database (optional):**
   ```bash
   cp /mnt/rectangularfile/pdf_index.db /mnt/rectangularfile/pdf_index.db.backup
   ```

3. **Remove the database:**
   ```bash
   rm /mnt/rectangularfile/pdf_index.db
   ```

4. **Important: Remove CalDAV settings** to avoid creating duplicate todos:
   - Either don't configure CalDAV environment variables, or
   - Set CalDAV to disabled in settings after first start

5. **Move archived files back to the upload folder** (if archiving is enabled):
   ```bash
   # Files are archived after processing — move them back to be re-scanned
   cp -r /mnt/rectangularfile/archive/* /mnt/onyx/
   ```

6. **Start the service:**
   ```bash
   sudo systemctl start rectangular-file
   ```

7. The system will automatically discover and process all PDFs in `UPLOAD_FOLDER`.

8. **After processing completes**, configure CalDAV in the web UI if desired. Only new action items after this point will create todos.


## 📖 Documentation

- [Installation Guide](installation.md)
