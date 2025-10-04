"""
RectangularFile Configuration Module

Centralizes all configuration settings with environment variable support.
All paths default to sensible development values but can be overridden for production.
"""
import os
from pathlib import Path


class Config:
    """Application configuration with environment variable support."""

    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-default-secret-key-change-this')
    APP_PASSWORD_HASH = os.environ.get('APP_PASSWORD_HASH', '')

    # File Storage Paths
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', '/mnt/onyx')

    # Database Configuration
    DATABASE_PATH = os.environ.get('DATABASE_PATH', '/mnt/rectangularfile/pdf_index.db')

    # Model Configuration
    MODEL_NAME = os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-VL-7B-Instruct')
    MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', '/mnt/rectangularfile/qwencache')

    # Debug and Logging
    DEBUG_IMAGES_DIR = os.environ.get('DEBUG_IMAGES_DIR', '/mnt/rectangularfile/debug_images')

    # Processing Configuration
    FILE_WATCHER_POLLING_INTERVAL = float(os.environ.get('POLLING_INTERVAL', '30.0'))

    # Flask Server Configuration
    FLASK_HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.environ.get('FLASK_PORT', '5000'))
    FLASK_DEBUG = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

    # Upload Limits
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload

    # Session Configuration
    SESSION_COOKIE_NAME = 'rectangularfile_session'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        directories = [
            cls.UPLOAD_FOLDER,
            Path(cls.DATABASE_PATH).parent,
            cls.MODEL_CACHE_DIR,
            cls.DEBUG_IMAGES_DIR
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls):
        """Validate critical configuration settings."""
        errors = []

        # Check for default secret key in production
        if cls.SECRET_KEY == 'your-default-secret-key-change-this':
            errors.append("SECRET_KEY is using default value - please set a secure key")

        # Check for password hash
        if not cls.APP_PASSWORD_HASH:
            errors.append("APP_PASSWORD_HASH is not set - authentication will not work")

        return errors

    @classmethod
    def print_config(cls):
        """Print current configuration (for debugging/verification)."""
        config_items = [
            ("Upload Folder", cls.UPLOAD_FOLDER),
            ("Database Path", cls.DATABASE_PATH),
            ("Model Cache", cls.MODEL_CACHE_DIR),
            ("Debug Images", cls.DEBUG_IMAGES_DIR),
            ("Model Name", cls.MODEL_NAME),
            ("Polling Interval", f"{cls.FILE_WATCHER_POLLING_INTERVAL}s"),
            ("Flask Host", cls.FLASK_HOST),
            ("Flask Port", cls.FLASK_PORT),
            ("Flask Debug", cls.FLASK_DEBUG),
            ("Secret Key Set", "Yes" if cls.SECRET_KEY != 'your-default-secret-key-change-this' else "No (using default)"),
            ("Password Hash Set", "Yes" if cls.APP_PASSWORD_HASH else "No"),
        ]

        print("\n" + "="*60)
        print("RectangularFile Configuration")
        print("="*60)
        for label, value in config_items:
            print(f"{label:20} : {value}")
        print("="*60 + "\n")


# Create a singleton instance for easy importing
config = Config()
