# RectangularFile

![image](https://github.com/user-attachments/assets/c168eea6-1992-473e-9427-39dbf52b12c9)

RectangularFile is a powerful open-source document management system designed for handwritten notes from e-ink tablets (Onyx Boox, reMarkable, Supernote, etc.) or any kind of device that can output PDFs of handwritten content, such as iPads or Samsung tablets. It uses locally deployed AI with no cloud services required to transcribe handwriting, detect annotations, and make everything searchable.

## ✨ Features

- **🤖 AI-Powered Handwriting Recognition** - Uses Qwen2.5-VL-7B for accurate handwriting transcription
- **📝 Annotation Detection** - Automatically detects and indexes:
  - ✅ Green boxed text (for todos/important items)
  - 🟨 Yellow highlighted text
- **🔍 Full-Text Search** - Search across all your handwritten notes with folder filtering
- **📁 Multi-Device Support** - Automatically organizes notes from multiple devices
- **☁️ Word Clouds** - Visualize common themes across your notes
- **✏️ In-Place Editing** - Fix transcription errors directly in the web interface
- **🔒 Simple Authentication** - Single-user login system

## 🚀 Quick Start

1. Clone the repository
2. Create a virtualenv in your repo directory: `python3 -m venv venv`
2. Install dependencies: `pip install -r requirements.txt`
3. Set up authentication (see below)
4. Set up your Gunicorn installation with a systemd unit file like the provided example rectangular-file.service, place in `/etc/systemd/system/`
5. Point your e-ink devices to sync PDFs to `/mnt/onyx` (or configured folder)
6. `systemctl start && systemctl enable` 

## 📋 Requirements

- Python 3.8+
- NVIDIA GPU with 16GB+ VRAM (for Qwen2.5-VL-7B)
- 50GB+ disk space for model cache
- System packages:
  - `poppler` (macOS: `brew install poppler`)
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

`SECRET_KEY=<generated_key> APP_PASSWORD_HASH=<generated_hash>`


## 📖 Documentation

- [Installation Guide](docs/installation.md)