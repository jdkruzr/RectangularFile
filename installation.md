# RectangularFile Installation Guide

This guide will walk you through setting up RectangularFile on your server.

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation Steps](#installation-steps)
- [Inference API Setup](#inference-api-setup)
- [Database Initialization](#database-initialization)
- [Systemd Service](#systemd-service)
- [Nginx Configuration](#nginx-configuration)
- [Device Setup](#device-setup)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+ or Debian 11+ (other Linux distros should work)
- **CPU**: 2+ cores
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**:
  - Space for document storage and archive (depends on your note volume)
  - SSD recommended
- **Python**: 3.8 or higher
- **Inference**: An OpenAI-compatible vision API endpoint (local or cloud — see [Inference API Setup](#inference-api-setup))

### Required System Packages

```bash
# Update package list
sudo apt update

# Install Python and development tools
sudo apt install -y python3-pip python3-venv python3-dev

# Install PDF processing dependencies
sudo apt install -y poppler-utils

# Install image processing libraries (required by Pillow/pdf2image)
sudo apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev

# Install git (if not already installed)
sudo apt install -y git
```

## Installation Steps

### 1. Create Application User

```bash
# Create a dedicated user for the application
# I like to use "sysop" because it was the username for
# the superuser on OpenVMS when I was given those
# creds at my first job in high school.
sudo useradd -r -s /bin/bash -d /home/sysop sysop
sudo mkdir -p /home/sysop
sudo chown sysop:sysop /home/sysop
```

### 2. Clone Repository

```bash
# Switch to application user
sudo su - sysop

# Clone the repository
git clone https://github.com/jdkruzr/RectangularFile.git
cd RectangularFile
```

### 3. Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Create Required Directories

```bash
# Application data directory
sudo mkdir -p /mnt/rectangularfile
sudo mkdir -p /mnt/rectangularfile/debug_images
sudo mkdir -p /mnt/rectangularfile/archive
sudo chown -R sysop:sysop /mnt/rectangularfile

# Document upload directory (or wherever your devices sync to)
sudo mkdir -p /mnt/onyx
sudo chown -R sysop:sysop /mnt/onyx
```

## Inference API Setup

RectangularFile requires an OpenAI-compatible vision API for handwriting recognition. It does not include a local inference engine — you point it at whichever backend you prefer.

### Option A: Local — vLLM

[vLLM](https://docs.vllm.ai/) is recommended for self-hosted inference. Qwen2.5-VL-7B works well for handwriting:

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000
```

Set in your service file:
```
Environment=INFERENCE_API_BASE=http://localhost:8000/v1
Environment=INFERENCE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
```

### Option B: Local — Ollama

[Ollama](https://ollama.com/) is the easiest local option:

```bash
ollama pull qwen2.5vl:7b
```

Set in your service file:
```
Environment=INFERENCE_API_BASE=http://localhost:11434/v1
Environment=INFERENCE_MODEL=qwen2.5vl:7b
```

### Option C: Cloud — OpenAI

```
Environment=INFERENCE_API_BASE=https://api.openai.com/v1
Environment=INFERENCE_API_KEY=sk-your-key-here
Environment=INFERENCE_MODEL=gpt-4o
```

Any vision-capable model works. GPT-4o and similar models handle handwriting well.

## Database Initialization

The database will be created automatically on first run at `/mnt/rectangularfile/pdf_index.db`.

To manually initialize or check the database:

```bash
# Initialize database
python db_migration_tool.py init

# Check current status
python db_migration_tool.py status
```

## Systemd Service

### 1. Create Service File

```bash
sudo nano /etc/systemd/system/rectangular-file.service
```

Add the following content (adjust paths and credentials as needed):

```ini
[Unit]
Description=RectangularFile Handwritten Note Manager
After=network.target

[Service]
Type=simple
User=sysop
Group=sysop
WorkingDirectory=/home/sysop/RectangularFile
Environment="PATH=/home/sysop/RectangularFile/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Authentication settings (CHANGE THESE!)
Environment="SECRET_KEY=your-generated-secret-key-here"
Environment="APP_PASSWORD_HASH=your-generated-password-hash-here"

# Inference API (required — point to your backend)
Environment="INFERENCE_API_BASE=http://localhost:8000/v1"
Environment="INFERENCE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct"
# Environment="INFERENCE_API_KEY=your-api-key"  # required for cloud providers

# File paths
Environment="UPLOAD_FOLDER=/mnt/onyx"
Environment="DATABASE_PATH=/mnt/rectangularfile/pdf_index.db"

# Archive (moves processed files out of upload folder to enable re-upload detection)
Environment="ARCHIVE_ENABLED=true"
Environment="ARCHIVE_FOLDER=/mnt/rectangularfile/archive"

ExecStart=/home/sysop/RectangularFile/venv/bin/gunicorn -w 1 -b 0.0.0.0:5000 --timeout 300 --access-logfile - --error-logfile - main:app
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 2. Generate Authentication Credentials

```bash
# Generate secret key
python -c 'import secrets; print(secrets.token_hex(32))'

# Generate password hash (replace 'yourpassword')
python -c "import hashlib; print(hashlib.sha256('yourpassword'.encode()).hexdigest())"
```

Update the service file with these values.

### 3. Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable rectangular-file

# Start the service
sudo systemctl start rectangular-file

# Check status
sudo systemctl status rectangular-file

# View logs
sudo journalctl -u rectangular-file -f
```

## Device Setup

### Onyx Boox Devices

1. Open **Settings** → **Apps** → **Push**
2. Set up WebDAV/FTP sync to your server's `/mnt/onyx` directory
3. Enable auto-upload for PDFs
4. Set sync schedule (recommended: every hour)

### Supernote Devices

1. Use Supernote Partner app
2. Configure sync folder to upload to your server

### reMarkable

1. Use rclone or rmapi to sync files
2. Set up automated sync via cron job

## Troubleshooting

### Inference API Not Responding

If documents are queued but not being processed:

1. Check that your inference backend is running and reachable:
   ```bash
   curl http://localhost:8000/v1/models
   ```
2. Verify `INFERENCE_API_BASE` is set correctly in the service file
3. Check logs for connection errors: `sudo journalctl -u rectangular-file -n 100`
4. If using a cloud API, confirm your `INFERENCE_API_KEY` is valid

### Permission Issues

Ensure correct ownership:
```bash
sudo chown -R sysop:sysop /mnt/rectangularfile
sudo chown -R sysop:sysop /mnt/onyx
sudo chown -R sysop:sysop /home/sysop/RectangularFile
```

### Database Lock Issues

If you see "database is locked" errors:
1. Stop the service: `sudo systemctl stop rectangular-file`
2. Check for stuck processes: `ps aux | grep python`
3. Restart the service: `sudo systemctl start rectangular-file`

### Files Not Being Processed

If files appear in the upload folder but aren't being picked up:
1. Check `UPLOAD_FOLDER` is set correctly
2. Verify the service user has read access to the folder
3. Check logs for file watcher errors: `sudo journalctl -u rectangular-file -n 50`
4. Use `python scan_filesystem.py` to compare what's on disk vs the database

## Next Steps

Once installation is complete:

1. Access the web interface at `http://your-server-ip:5000`
2. Log in with the password you configured
3. Upload a test PDF to verify OCR functionality
4. Configure your e-ink devices to sync
5. Optionally configure CalDAV in the web UI for todo sync

## Support

If you encounter issues:

1. Check the logs: `sudo journalctl -u rectangular-file -n 100`
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Open an issue on GitHub with error details
