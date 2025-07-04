# RectangularFile Installation Guide

This guide will walk you through setting up RectangularFile on your server.

## Table of Contents
- [System Requirements](#system-requirements)
- [GPU Setup](#gpu-setup)
- [Installation Steps](#installation-steps)
- [Model Setup](#model-setup)
- [Database Initialization](#database-initialization)
- [Systemd Service](#systemd-service)
- [Nginx Configuration](#nginx-configuration)
- [Device Setup](#device-setup)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+ or Debian 11+ (other Linux distros should work)
- **CPU**: 4+ cores recommended
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 3060 Ti 16GB, RTX 4060 Ti 16GB, etc.)
- **Storage**: 
  - 50GB for model cache
  - Additional space for document storage
  - SSD strongly recommended
- **Python**: 3.8 or higher

### Required System Packages

```bash
# Update package list
sudo apt update

# Install Python and development tools
sudo apt install -y python3-pip python3-venv python3-dev

# Install PDF processing dependencies
sudo apt install -y poppler-utils

# Install image processing libraries
sudo apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Install git (if not already installed)
sudo apt install -y git
```

## GPU Setup

### 1. Install NVIDIA Drivers

```bash
# Check if you have NVIDIA drivers installed
nvidia-smi

# If not installed, install them:
sudo apt install nvidia-driver-535  # or latest version

# Reboot after installation
sudo reboot
```

### 2. Install CUDA Toolkit

RectangularFile requires CUDA for GPU acceleration:

```bash
# Install CUDA 12.1 (or compatible version)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
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
# Install PyTorch with CUDA support for 5xxx series cards
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt
```

### 5. Create Required Directories

```bash
# Create directories for application data
sudo mkdir -p /mnt/rectangularfile
# Or wherever you want to put e.g. the RF sqlite3 database.
sudo mkdir -p /mnt/onyx  # Or your preferred document upload directory
# I created a mount for the directory that is created by Boox devices
# automatically.
sudo chown -R sysop:sysop /mnt/rectangularfile
sudo chown -R sysop:sysop /mnt/onyx

# Create cache directory for models
mkdir -p /mnt/rectangularfile/qwencache
mkdir -p /mnt/rectangularfile/debug_images
```

## Model Setup

The Qwen2.5-VL-7B model will be downloaded automatically on first run, but you can pre-download it:

```bash
# Activate virtual environment
source venv/bin/activate

# Pre-download the model
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', cache_dir='/mnt/rectangularfile/qwencache')"
```

**Note**: This will download approximately 15GB of model files.

## Database Initialization

The database will be created automatically on first run at `/mnt/rectangularfile/pdf_index.db`.

To manually initialize or check the database:

```bash
# Run the schema manager
python -c "from db.schema_manager import SchemaManager; sm = SchemaManager('/mnt/rectangularfile/pdf_index.db'); sm.initialize_database()"
```

## Systemd Service

### 1. Create Service File

```bash
sudo nano /etc/systemd/system/rectangular-file.service
```

Add the following content:

```ini
[Unit]
Description=RectangularFile Handwritten Note Manager
After=network.target

# Change the below as necessary.
[Service]
Type=simple
User=sysop
Group=sysop
WorkingDirectory=/home/sysop/RectangularFile
Environment="PATH=/home/sysop/RectangularFile/venv/bin:/usr/local/cuda-12.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64"
Environment="TRANSFORMERS_CACHE=/mnt/rectangularfile/qwencache"
Environment="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

# Authentication settings (CHANGE THESE!)
Environment="SECRET_KEY=your-generated-secret-key-here"
Environment="APP_PASSWORD_HASH=your-generated-password-hash-here"

ExecStart=/home/sysop/RectangularFile/venv/bin/gunicorn -w 1 -b 0.0.0.0:8000 --timeout 300 --access-logfile - --error-logfile - main:app
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

### GPU Memory Issues

If you encounter CUDA out of memory errors:

1. Check GPU memory usage: `nvidia-smi`
2. Ensure no other processes are using GPU
3. Try restarting the service: `sudo systemctl restart rectangular-file`

### Model Download Issues

If model download fails:

1. Check disk space: `df -h`
2. Check network connectivity
3. Try manual download with resume:
   ```bash
   cd /mnt/rectangularfile/qwencache
   wget -c https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/model.safetensors
   ```

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

## Next Steps

Once installation is complete:

1. Access the web interface at `http://your-server-ip`
2. Log in with the password you configured
3. Upload a test PDF to verify OCR functionality
4. Configure your e-ink devices to sync
5. Check the [Configuration Guide](configuration.md) for advanced settings

## Support

If you encounter issues:

1. Check the logs: `sudo journalctl -u rectangular-file -n 100`
2. Verify GPU is detected: `nvidia-smi`
3. Ensure all dependencies are installed
4. Open an issue on GitHub with error details