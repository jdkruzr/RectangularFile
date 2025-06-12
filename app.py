# app.py
"""
Backward compatibility for the new modular structure.
This file redirects to the new application entry point.
"""

from main import app

# This allows gunicorn to still use "app:app" as the entry point